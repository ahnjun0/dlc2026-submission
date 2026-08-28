"""vLLM 배치 추론: 문항당 n개 샘플을 생성해 jsonl로 저장한다. (GPU 환경 전용)

사용 (GPU 박스에서):
    python src/inference/generate.py --input data/raw/deep_chal_math_leaderboard.csv \
        --output experiments/exp-001_baseline/gens.jsonl \
        --model Qwen/Qwen2.5-3B-Instruct --n 32

출력 jsonl 한 줄 = {"id": ..., "samples": [text, ...]}
greedy 측정은 --n 1 --temperature 0 으로.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REVISE_PROMPT = (
    "Re-examine the solution above. Work through the problem again from the beginning, "
    "independently of your previous reasoning, and check whether the earlier answer holds. "
    "If you find an error, give the corrected solution. If the earlier answer was right, confirm it. "
    "The final answer is always an integer. "
    "End with the final answer in the form \\boxed{integer}."
)

SYSTEM_PROMPT = (
    "You are an expert competition mathematician. Solve the problem step by step. "
    "The final answer is always an integer. "
    "End your solution with the final answer in the form \\boxed{integer}."
)


def build_prompts(df: pd.DataFrame, tokenizer, no_think: bool = False,
                  prior: dict | None = None, revise_instr: str | None = None,
                  system_prompt: str | None = None,
                  revise_style: str = "assistant",
                  assistant_prefill: str | None = None,
                  prefill_col: str | None = None) -> list[str]:
    """`no_think=True`면 Qwen3 계열의 **사고 모드를 끈다**.

    Qwen3의 chat template은 기본이 thinking ON이라, 끄지 않으면 `<think>` 블록으로
    출력이 몇 배 길어진다. 우리는 **짧은 teacher**를 쓰려고 8B를 고른 것이므로
    (API 파일럿 실측 중앙값 2,864자) 사고 모드를 켜면 고른 이유 자체가 사라진다.
    Qwen2.5 계열 템플릿은 이 kwarg를 무시하므로 그냥 둬도 무해하다."""
    kw = {"enable_thinking": False} if no_think else {}
    prompts = []
    for row in df.itertuples():
        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": row.question},
        ]
        if prior is not None and revise_style == "user":
            # **틀린 풀이를 user 턴 안에 넣는다.**
            # assistant 턴으로 넣으면 SFT 때 그 턴도 loss에 들어가 **오답 생성을 학습**한다
            # (TRL의 assistant_only_loss는 assistant 턴 '전부'를 학습하므로 못 막는다).
            # user 턴에 넣으면 3-메시지 구조가 되어 기존 sft_lora.py가 그대로 돌고,
            # loss는 교정 풀이에만 걸린다. 학습·추론 형식이 같아야 하므로 생성도 이 형식으로 한다.
            messages[-1]["content"] = (
                f"{row.question}\n\n[A previous attempt]\n{prior.get(row.id, '')}\n\n"
                f"{revise_instr or REVISE_PROMPT}"
            )
        elif prior is not None:
            # **2턴 교정 모드** — 모델 자신의 앞선 풀이를 assistant 턴으로 되먹인다.
            #
            # exp-010(rationalization)이 죽은 이유는 **학습/추론 불일치**였다: 힌트를 받고 푼
            # 풀이를 학습시켰는데 추론 시엔 힌트가 없어, 스스로 유도 못 하는 결론을 흉내 내게 됐다
            # (타깃 아닌 주제까지 −6.7%p 확산). 2턴 형태는 그 병이 구조적으로 없다 —
            # **자기 오답은 추론 시에도 반드시 존재**하기 때문이다(어차피 32샘플을 뽑는다).
            messages.append({"role": "assistant", "content": prior.get(row.id, "")})
            messages.append({"role": "user", "content": revise_instr or REVISE_PROMPT})
        p = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, **kw)
        pre = (str(getattr(row, prefill_col)) if prefill_col else assistant_prefill)
        if pre:
            # chat template 이 만든 generation prompt 뒤에 그대로 이어 붙이면
            # 모델은 이 문자열 **다음부터** 생성한다 = 준수율 100%.
            p = p + pre
        prompts.append(p)
    return prompts


def conf_stats(comp, window: int = 128) -> dict:
    """샘플 하나의 신뢰도 지표. **토큰 단위 값은 저장하지 않고 스칼라로 요약한다.**

    · `mean_ent`  : 궤적 평균 토큰 엔트로피 (엔트로피 가중 투표용, arXiv:2511.02309 계열)
    · `min_win_lp`: 128토큰 창 평균 로그확률의 **최솟값** (DeepConf의 group confidence —
                    궤적 전체는 자신 있는데 한 구간에서 무너지는 경우를 잡는다)
    · `min_win_ent`: 같은 창의 엔트로피 최댓값(= 신뢰도 최솟값)

    우리가 이미 닫은 축(21: 답 토큰 평균 logprob 가중)과 **다른 신호**를 노린다.
    """
    lps = comp.logprobs
    if not lps:
        return {}
    ents, sel = [], []
    for step in lps:
        vals = [v.logprob for v in step.values()]
        if not vals:
            continue
        m = max(vals)
        ps = [pow(2.718281828459045, v - m) for v in vals]
        z = sum(ps) or 1.0
        ps = [x / z for x in ps]
        ents.append(-sum(x * (x and __import__("math").log(x)) for x in ps))
        sel.append(m)
    if not ents:
        return {}
    n = len(ents)
    w = min(window, n)
    win_lp = [sum(sel[i:i + w]) / w for i in range(0, max(1, n - w + 1))]
    win_ent = [sum(ents[i:i + w]) / w for i in range(0, max(1, n - w + 1))]
    return {"n_tok": n,
            "mean_ent": round(sum(ents) / n, 5),
            "min_win_lp": round(min(win_lp), 5),
            "max_win_ent": round(max(win_ent), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=32, help="문항당 샘플 수")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.0,
                    help="누적 확률이 아니라 **최빈 토큰 대비 상대 확률**로 자른다. "
                         "AIMO3 3팀이 독립 수렴한 설정은 T=1.0 + min_p=0.02 + top_p 미설정. "
                         "우리 스윕(exp-007)은 온도 0.7~0.9만 봤고 min_p는 미탐색")
    ap.add_argument("--top-k", type=int, default=-1, help="-1 = 미사용")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42,
                    help="**vLLM은 n>1일 때 자식 시드를 seed, seed+1, ..., seed+n-1로 만든다.** "
                         "따라서 seed 42와 44는 n=32에서 32개 중 30개가 겹친다 "
                         "(2026-08-13 실측: 정렬 오프셋에서 완전 동일 텍스트 25.15% vs 비정렬 0.15%). "
                         "**독립 반복을 원하면 시드 간격을 n 이상으로 벌릴 것** — "
                         "예: 42 / 1042 / 2042. 종전 s42/s44/s46/s48은 독립 4회가 아니었다")
    ap.add_argument("--limit", type=int, default=0, help="앞 N문항만 (스모크 테스트용)")
    ap.add_argument("--lora", default=None, help="LoRA 어댑터 경로 (베이스 위에 적용)")
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="vLLM VRAM 몫. **다른 학습과 GPU 를 공유할 때 필수** — 기본 0.9 는 "
                         "카드 전체를 잡아 동거 중인 학습을 죽인다 (2026-08-24 PRO 6000 실측)")
    ap.add_argument("--quantization", default=None,
                    help="vLLM 양자화 방식 (fp8 등). **규칙 4.2 명시 허용**. 미지정 시 bf16")
    ap.add_argument("--revise-from", default=None,
                    help="2턴 교정 모드: 앞선 생성 jsonl. 각 문항의 --revise-index 번째 샘플을 "
                         "assistant 턴으로 되먹이고 재풀이를 생성한다")
    ap.add_argument("--revise-index", type=int, default=0, help="되먹일 샘플 번호")
    ap.add_argument("--revise-instr", default=None,
                    help="재검토 지시문 오버라이드. **교차 재풀이에서는 '어느 쪽이 맞는지 고르라'가 아니라 "
                         "'결론을 신뢰하지 말고 처음부터 독립적으로 다시 풀라'여야 한다** — "
                         "선택기는 이미 3종 전부 동전이었다(52.6/50.0/56.1%)")
    ap.add_argument("--revise-style", choices=["assistant", "user"], default="assistant",
                    help="user = 앞선 풀이를 user 턴에 embed (학습 형식과 일치, loss 마스킹 불필요)")
    ap.add_argument("--no-think", action="store_true",
                    help="Qwen3 계열의 사고 모드를 끈다 (Qwen2.5에는 무영향)")
    ap.add_argument("--system-prompt", default=None, help="SYSTEM_PROMPT 오버라이드 (프롬프트 스윕용)")
    ap.add_argument("--prefill-col", default=None,
                    help="입력 CSV의 이 컬럼을 **행마다 다른** assistant 접두로 쓴다 "
                         "(Math-Shepherd식 prefix 롤아웃용). --assistant-prefill 보다 우선. "
                         "기본 None = 기존 동작 불변")
    ap.add_argument("--assistant-prefill", default=None,
                    help="assistant 턴을 이 문자열로 **시작시킨다** (준수율 100% 강제용). "
                         "프롬프트 끝에 붙이고, 저장 시 각 샘플 앞에 되돌려 붙여 "
                         "downstream 분석이 완전한 풀이를 보게 한다. 기본 None = 기존 동작 불변")
    ap.add_argument("--logprobs", action="store_true", help="샘플별 평균 로그확률 기록 (확신도 가중 투표 분석용)")
    ap.add_argument("--chunk", type=int, default=250,
                    help="이 문항 수마다 디스크에 기록 (0=전량 한 번에). 장애 시 손실 상한이 된다")
    ap.add_argument("--resume", action="store_true",
                    help="출력 파일에 이미 있는 문항은 건너뛰고 이어서 생성")
    args = ap.parse_args()

    if args.system_prompt:
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = args.system_prompt

    from vllm import LLM, SamplingParams  # GPU 환경에서만 import

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    # --- 2턴 교정 모드: 앞선 풀이를 문항별로 싣는다 ---
    prior = None
    if args.revise_from:
        prior = {}
        for line in open(args.revise_from):
            r = json.loads(line)
            sams = r.get("samples") or []
            if sams:
                prior[r["id"]] = sams[min(args.revise_index, len(sams) - 1)]
        before = len(df)
        df = df[df.id.isin(prior)].reset_index(drop=True)
        print(f"[revise] 앞선 풀이 {len(prior):,}건 · 대상 문항 {before} -> {len(df)}", flush=True)
        assert len(df), "revise-from과 input의 id가 하나도 안 겹친다"  # 체크리스트 14

    # --- resume: 이미 끝난 문항은 건너뛴다 ---
    # 종전에는 전량 생성 후 한 번에 기록해서, 4~6시간짜리 런이 장애 한 번에 통째로
    # 사라졌다 (D-Day 831문항 추론의 최대 운영 리스크). 청크 단위로 append 한다.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _digest(path: str | None) -> str | None:
        """디렉터리·파일의 내용 해시. **크기로 대체하지 않는다** — 같은 크기의
        다른 어댑터를 구분하지 못하면 manifest가 무의미해진다 (외부 리뷰 지적)."""
        if not path or not Path(path).exists():
            return None
        h = hashlib.sha256()
        p = Path(path)
        files = sorted(f for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file())
        for fp in files:
            h.update(fp.name.encode())
            with fp.open("rb") as fh:  # 스트리밍 — 대용량도 내용 전체를 읽는다
                for block in iter(lambda: fh.read(8 << 20), b""):
                    h.update(block)
        return h.hexdigest()[:16]

    def _run_config() -> dict:
        return {
            "model": args.model, "lora": args.lora, "lora_digest": _digest(args.lora),
            "input": args.input, "input_digest": _digest(args.input),
            "n": 1 if args.temperature == 0 else args.n,
            "temperature": args.temperature, "top_p": args.top_p,
            "max_tokens": args.max_tokens, "seed": args.seed,
            "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
        }

    mf_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    cfg_now = _run_config()

    done: set = set()
    if args.resume and out_path.exists():
        # **설정 불일치 resume 차단**: 다른 어댑터·시드·온도로 이어붙이면 한 파일에
        # 서로 다른 설정의 샘플이 섞이고 manifest는 마지막 것만 남는다 (외부 리뷰 지적).
        if mf_path.exists():
            prev = json.loads(mf_path.read_text()).get("config", {})
            diff = {k: (prev.get(k), v) for k, v in cfg_now.items() if prev.get(k) != v}
            if diff:
                raise SystemExit(
                    "[resume 중단] 기존 생성물과 설정이 다릅니다. 이어붙이면 샘플이 섞입니다.\n"
                    + "\n".join(f"  {k}: 기존={a!r} → 현재={b!r}" for k, (a, b) in diff.items())
                    + f"\n  (의도한 재생성이라면 {out_path} 와 manifest를 먼저 지우십시오)"
                )
        else:
            print("[resume] 기존 manifest가 없어 설정 일치를 검증하지 못했습니다 — 주의")
        # 손상·불완전 레코드는 **버리고 파일을 잘라 다시 쓴다**.
        # (외부 리뷰 지적) 종전에는 깨진 줄을 남긴 채 뒤에 append 해서 파일이
        # 영구 손상됐고, id만 보고 완료 판정해 **샘플이 n개 미만인 레코드도 skip**했다.
        kept: list[str] = []
        n_bad = 0
        expected_n = 1 if args.temperature == 0 else args.n
        with out_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if len(rec.get("samples", [])) < expected_n:
                        n_bad += 1
                        continue  # 표가 모자란 레코드는 다시 생성
                    if rec["id"] in done:
                        n_bad += 1
                        continue  # 중복 id는 조용히 덮어쓰지 않고 하나만 남긴다
                    done.add(rec["id"])
                    kept.append(line if line.endswith("\n") else line + "\n")
                except Exception:
                    n_bad += 1  # 잘린 마지막 줄 등
        with out_path.open("w") as f:  # 정상 레코드만으로 재작성
            f.writelines(kept)
        before = len(df)
        df = df[~df.id.isin(done)]
        print(f"[resume] 유효 {len(done):,}문항 건너뜀 / 폐기 {n_bad:,}행 / 남은 {len(df):,} (전체 {before:,})")
        if df.empty:
            print("[resume] 이미 전부 완료됨")
            return

    # LoRA 버퍼가 VRAM을 추가로 먹으므로 KV 캐시 몫을 줄인다 (24GB 기준)
    lora_kwargs = (
        {"enable_lora": True, "max_lora_rank": 128, "gpu_memory_utilization": 0.82}
        if args.lora
        else {}
    )
    if args.gpu_mem_util is not None:
        lora_kwargs["gpu_memory_utilization"] = args.gpu_mem_util
        print(f"**VRAM 몫 {args.gpu_mem_util}** (GPU 공유 모드)")
    # **양자화** (exp-099) — 규칙 4.2 가 명시 허용하는데 실험 0회였던 축.
    # 같은 가중치의 미세 섭동이 파트너급 다양성을 만드는지 본다. 학습 비용 0.
    if args.quantization:
        lora_kwargs["quantization"] = args.quantization
        print(f"**양자화: {args.quantization}**")
    llm = LLM(model=args.model, seed=args.seed, dtype="bfloat16", **lora_kwargs)
    tokenizer = llm.get_tokenizer()

    greedy = args.temperature == 0
    sp = SamplingParams(
        n=1 if greedy else args.n,
        temperature=args.temperature,
        top_p=1.0 if greedy else args.top_p,
        min_p=0.0 if greedy else args.min_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        # top-5를 받아야 **토큰 엔트로피**를 계산할 수 있다 (DeepConf/엔트로피 가중 검증용).
        # 0이면 선택 토큰 확률만 나와서 엔트로피를 못 구한다.
        logprobs=5 if args.logprobs else None,
    )
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("adapter", 1, args.lora)

    mode = "a" if (args.resume and done) else "w"
    n_written = 0
    chunk = args.chunk if args.chunk > 0 else len(df)
    with out_path.open(mode) as f:
        for start in range(0, len(df), chunk):
            part = df.iloc[start:start + chunk]
            outputs = llm.generate(
                build_prompts(part, tokenizer, no_think=args.no_think,
                              prior=prior, system_prompt=args.system_prompt,
                              revise_style=args.revise_style,
                              revise_instr=args.revise_instr, assistant_prefill=args.assistant_prefill,
                              prefill_col=args.prefill_col),
                sp, lora_request=lora_req)
            for row_id, out in zip(part.id, outputs):
                if args.prefill_col:
                    pre = str(part.loc[part.id == row_id, args.prefill_col].iloc[0])
                else:
                    pre = args.assistant_prefill or ""
                rec = {"id": row_id, "samples": [pre + c.text for c in out.outputs]}
                if args.logprobs:
                    rec["mean_logprobs"] = [
                        (c.cumulative_logprob / max(1, len(c.token_ids))) if c.cumulative_logprob is not None else None
                        for c in out.outputs
                    ]
                    rec["conf"] = [conf_stats(c) for c in out.outputs]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # 전원 장애 대비 — flush만으로는 OS 버퍼에 남는다
            n_written += len(part)
            print(f"[chunk] {n_written:,}/{len(df):,} 기록 완료", flush=True)
    print(f"wrote {n_written} problems x {sp.n} samples -> {out_path}")

    # --- manifest: 재현·검증용 실행 지문. config는 resume 시 일치 검증에 쓰인다 ---
    mf_path.write_text(json.dumps({
        "output": str(out_path), "config": cfg_now,
        "problems_written": n_written, "resumed_from": len(done),
    }, ensure_ascii=False, indent=2))
    print(f"manifest -> {mf_path}")


if __name__ == "__main__":
    main()

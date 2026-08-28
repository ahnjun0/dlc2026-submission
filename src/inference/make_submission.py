"""생성 jsonl → 다수결/풀링/캐스케이드 → submission.csv. 로컬(M4)에서 실행 가능.

D-Day 하니스. 문서상 전략(풀링·캐스케이드)이 코드에 없어 2026-08-06에 보강했다.

사용 예:
  # 단일 모델
  python src/inference/make_submission.py --gens gens.jsonl --output sub.csv

  # 풀링 (첫 번째가 주 모델 = 동률 우선권)
  python src/inference/make_submission.py \
      --gens v2.jsonl moonshot.jsonl --output sub.csv

  # 풀링 + 모델별 표 수 지정 (v2는 64표, 파트너는 32표)
  python src/inference/make_submission.py \
      --gens v2.jsonl moonshot.jsonl --k 64 32 --output sub.csv

  # 캐스케이드: 주 모델 최다표가 임계 이하인 문항만 파트너 표를 증원
  python src/inference/make_submission.py \
      --gens v2.jsonl moonshot.jsonl --cascade-threshold 14 --output sub.csv

  # 검증 모드 (정답 CSV로 maj@k 곡선)
  python src/inference/make_submission.py --gens gens.jsonl --eval val.csv

  # 제출 전 ID 정합 강제 (기대 ID 목록과 완전 일치해야 통과)
  python src/inference/make_submission.py --gens ... --output sub.csv \
      --expect-ids data/raw/deep_chal_math_leaderboard_filtered.csv
"""

import argparse
import json
from collections import Counter
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.parse import extract_answer  # noqa: E402
from src.eval.score import maj_at_k, majority_vote, pooled_vote  # noqa: E402

FALLBACK_ANSWER = 0  # 파싱 전멸 시 빈 값(무조건 오답) 대신 넣는 값


def load_gens(path: str, require_k: int = 0) -> dict[str, list[int | None]]:
    """생성물 로드. **중복 ID는 즉시 실패**하고, require_k>0이면 표 수를 강제한다.

    종전에는 중복 ID를 마지막 레코드로 조용히 덮어쓰고, `--k 32`인데 표가 8개뿐이어도
    8표로 진행했다 (외부 리뷰 지적). 제출 직전에 조용히 표가 줄어드는 것은
    점수를 갉아먹으면서 원인도 남기지 않는다.
    """
    parsed: dict[str, list[int | None]] = {}
    short: list[tuple[str, int]] = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["id"]
            if pid in parsed:
                raise SystemExit(f"[중단] {Path(path).name}에 중복 ID: {pid}")
            samples = rec["samples"]
            if require_k and len(samples) < require_k:
                short.append((pid, len(samples)))
            parsed[pid] = [extract_answer(s) for s in samples]
    if short:
        head = ", ".join(f"{p}({k})" for p, k in short[:5])
        raise SystemExit(
            f"[중단] {Path(path).name}: 표가 {require_k}개 미만인 문항 {len(short)}건 — {head}"
        )
    return parsed


def build_groups(sources: list[dict[str, list[int | None]]], pid: str,
                 ks: list[int] | None) -> list[list[int | None]]:
    """문항 하나에 대해 모델별 표 묶음을 만든다 (우선순위 = 인자 순서)."""
    groups = []
    for idx, src in enumerate(sources):
        preds = src.get(pid, [])
        if ks and idx < len(ks) and ks[idx] > 0:
            preds = preds[: ks[idx]]
        groups.append(preds)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True,
                    help="생성 jsonl (여러 개면 풀링. **첫 번째가 주 모델** = 동률 우선권)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--k", nargs="*", type=int, default=None,
                    help="모델별 사용할 표 수 (미지정=전부). 예: --k 64 32")
    ap.add_argument("--cascade-threshold", type=int, default=0,
                    help=">0이면 주 모델 최다표가 이 값 이하인 문항에만 파트너 표를 증원 "
                         "(교체가 아니라 증원 — 실측상 교체는 -1.7%p, 증원은 +1.07%p)")
    ap.add_argument("--eval", default=None, help="정답 CSV 경로 (검증 모드)")
    ap.add_argument("--expect-ids", default=None,
                    help="기대 ID가 담긴 CSV. 주면 누락/초과를 assert로 막는다")
    ap.add_argument("--restrict-to-expect", action="store_true",
                    help="--expect-ids에 없는 문항을 **명시적으로** 제외한다. "
                         "구 leaderboard(1,000문항) 생성물로 재편된 831문항에 제출할 때처럼, "
                         "초과분이 정당한 경우에만 쓸 것. 제외 건수를 반드시 출력한다")
    args = ap.parse_args()

    ks = args.k or []
    sources = [load_gens(p, ks[i] if i < len(ks) else 0)
               for i, p in enumerate(args.gens)]
    primary = sources[0]
    all_ids = list(primary)
    if args.restrict_to_expect:
        assert args.expect_ids, "--restrict-to-expect 는 --expect-ids 와 함께 써야 한다"
        want = set(pd.read_csv(args.expect_ids).id)
        dropped = [i for i in all_ids if i not in want]
        all_ids = [i for i in all_ids if i in want]
        print(f"[restrict] 기대 목록 밖 {len(dropped):,}문항 제외 → {len(all_ids):,}문항 "
              f"(예: {dropped[:3]})")
    print(f"입력 {len(sources)}개 / 주 모델 문항 {len(all_ids):,}")
    for p, s in zip(args.gens, sources):
        missing = len(set(all_ids) - set(s))
        print(f"  {Path(p).name:<40} 문항 {len(s):,}" + (f"  (주 모델 대비 누락 {missing:,})" if missing else ""))

    if args.eval:
        gold_df = pd.read_csv(args.eval)
        # 누락 ID를 조용히 분모에서 빼면 점수가 부풀려진다 (외부 리뷰 지적).
        # 빼되 **반드시 경고**하고, 누락이 크면 비교 자체를 신뢰하지 말 것.
        n_all = len(gold_df)
        gold_df = gold_df[gold_df.id.isin(primary)]
        if len(gold_df) < n_all:
            print(f"⚠ 평가 대상 {n_all:,}문항 중 생성물에 없는 {n_all - len(gold_df):,}문항을 "
                  f"분모에서 제외함 — 다른 실행과 비교할 때 분모가 같은지 확인할 것")
        golds = gold_df.answer.tolist()
        if len(sources) == 1:
            samples = [primary[i] for i in gold_df.id]
            n = max(len(s) for s in samples)
            print(f"problems={len(golds)}, samples/problem={n}")
            fail = sum(p is None for s in samples for p in s) / sum(len(s) for s in samples)
            print(f"parse failure rate: {fail:.2%}")
            for k in [1, 2, 4, 8, 16, 32, 64]:
                if k <= n:
                    print(f"maj@{k:>2}: {maj_at_k(samples, golds, k):.4f}")
        else:
            preds = [pooled_vote(build_groups(sources, i, args.k)) for i in gold_df.id]
            acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
            print(f"pooled({len(sources)} models): {acc:.4f}  (문항 {len(golds):,})")
        return

    assert args.output, "--output 또는 --eval 중 하나는 필요"

    rows, n_fallback, n_delegated = [], 0, 0
    for pid in all_ids:
        groups = build_groups(sources, pid, args.k)
        if args.cascade_threshold > 0 and len(groups) > 1:
            # 주 모델이 충분히 확신하면 파트너를 부르지 않는다 (생성비 절감)
            # **pandas를 쓰지 않는다**: 거대정수 환각(6,384자리 실측)이 섞이면
            # pd.Series가 float 변환을 시도하다 OverflowError로 죽는다.
            # 파서에 상한을 풀어도 집계 쪽에 같은 병이 남아 있었다 (2026-08-09 리허설이 적발).
            main_valid = [p for p in groups[0] if p is not None]
            top = max(Counter(main_valid).values()) if main_valid else 0
            if top > args.cascade_threshold:
                groups = groups[:1]
            else:
                n_delegated += 1
        vote = pooled_vote(groups) if len(groups) > 1 else majority_vote(groups[0])
        if vote is None:
            vote = FALLBACK_ANSWER
            n_fallback += 1
        # 거대 정수(int64 초과 — 확실한 오답이지만 파이프라인은 죽지 않아야 함)를
        # 포함해 모든 답을 문자열로 직렬화 (CSV 상으로는 동일한 정수 표기)
        rows.append({"id": pid, "answer": str(int(vote))})
    sub = pd.DataFrame(rows, dtype=str)

    # --- 제출 전 자가 검증 (하나라도 깨지면 파일을 쓰지 않는다) ---
    assert sub.answer.map(lambda x: bool(re.fullmatch(r"-?\d+", x))).all(), "정수가 아닌 답 존재"
    assert sub.id.is_unique, "중복 ID"
    assert sub.answer.notna().all() and (sub.answer.str.len() > 0).all(), "빈 답 존재"
    if args.expect_ids:
        want = set(pd.read_csv(args.expect_ids).id)
        have = set(sub.id)
        assert not (want - have), f"누락 ID {len(want - have)}건: {sorted(want - have)[:5]}"
        assert not (have - want), f"기대 밖 ID {len(have - want)}건: {sorted(have - want)[:5]}"
        assert len(sub) == len(want), f"행 수 불일치 {len(sub)} != {len(want)}"
        print(f"ID 정합 검증 통과: {len(want):,}행 완전 일치")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    msg = f"wrote {len(sub)} rows -> {out} (fallback {n_fallback}"
    if args.cascade_threshold > 0:
        msg += f", 캐스케이드 위임 {n_delegated}/{len(all_ids)}"
    print(msg + ")")


if __name__ == "__main__":
    main()

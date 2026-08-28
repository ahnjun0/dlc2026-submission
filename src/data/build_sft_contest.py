"""경시 전용 STaR 데이터 빌드 (exp-035).

v2 레시피를 **문제 풀만 바꿔** 적용한다: 자기 생성 → 엄격 검증 → 해결률 역비례 채택.
`build_sft_external.py`와 로직은 같으나 exp-024 재현성을 위해 그 파일은 손대지 않고
분리했고, 이 실험에만 필요한 두 가지를 추가한다.

추가 ①  **출처별 층화** — 경시 풀은 출처순으로 정렬돼 있어(앞 4/5가 olympiads,
        마지막 1/5이 aops_forum·cn_contest) **앞에서 자르면 frontier 밀도가 가장 높은
        슬라이스가 통째로 잘려나간다**(aops 11.3% / cn_contest 9.2%). 상한을 둘 때는
        반드시 출처 비례로 뽑는다. (기본은 상한 없음 — 전량 사용)

추가 ②  **frontier 독립 시드 재확인** — 8샘플 중 1개만 gold와 맞은 항목은 가장 값진
        층이면서 가장 위험하다. 그 답의 절반이 |20| 이하 소수이고 모델은 6가지 답으로
        흩어져 있어(2026-08-10 실측) **틀린 풀이가 우연히 정답 숫자에 도달**했을 수 있다.
        다른 시드 생성물을 주면, 정답이 재현된 frontier 항목만 채택한다.

혼합 금지: `--base`가 없다 — 경시 데이터만 쓴다. mix 실측에서 v2 데이터로 1:4 희석했더니
φ가 0.726→0.878로 올라 파트너 가치가 절반이 됐다. **희석은 다양성을 죽인다.**

사용:
  python src/data/build_sft_contest.py \
      --gens experiments/exp-035_contest_star/contest_samp8_s42.jsonl \
      --recheck-gens experiments/exp-035_contest_star/frontier_samp8_s43.jsonl \
      --output data/processed/sft_contest.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.parse import extract_answer  # noqa: E402
from src.eval.strict_verify import is_trainable, strict_answer  # noqa: E402
from src.inference.generate import SYSTEM_PROMPT  # noqa: E402

HANGUL = re.compile(r"[가-힣]")


def pick_count(rate: float) -> int:
    """해결률 역비례 채택 — v2 원본 규칙 (easy 1 / mid 2 / frontier 4)."""
    if rate >= 0.6:
        return 1
    if rate >= 0.25:
        return 2
    return 4


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", str(q)).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--problems", default=str(ROOT / "data/processed/contest_pool.csv"))
    ap.add_argument("--recheck-gens", default=None,
                    help="frontier 항목의 독립 시드 생성물. 주면 정답 재현된 것만 채택")
    ap.add_argument("--output", default=str(ROOT / "data/processed/sft_contest.jsonl"))
    ap.add_argument("--max-records", type=int, default=0,
                    help="0=전량. >0이면 **출처 비례 층화**로 줄인다 (앞에서 자르지 않는다)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    prob = pd.read_csv(args.problems)
    gold = {r.id: int(r.answer) for r in prob.itertuples()}
    question = dict(zip(prob.id, prob.question))
    source = dict(zip(prob.id, prob.source))

    # 절대 규칙 6 재확인 (입력 단계에서 이미 제거했으나 빌드 시점에 한 번 더)
    lb_norm = {normalize(q) for q in pd.read_csv(
        ROOT / "data/raw/deep_chal_math_leaderboard_filtered.csv").question}
    leaked = {pid for pid, q in question.items() if normalize(q) in lb_norm}
    print(f"[dedup] leaderboard 일치 {len(leaked)}건" + (" — 제외" if leaked else " (재확인 통과)"))

    samples: dict[str, list[str]] = {}
    for line in open(args.gens):
        r = json.loads(line)
        samples.setdefault(r["id"], []).extend(r["samples"])
    print(f"샘플 보유 문항: {len(samples):,}")

    recheck: dict[str, list[str]] = {}
    if args.recheck_gens and Path(args.recheck_gens).exists():
        for line in open(args.recheck_gens):
            r = json.loads(line)
            recheck.setdefault(r["id"], []).extend(r["samples"])
        print(f"frontier 재확인 생성물: {len(recheck):,}문항")

    pool: dict[str, list[tuple[str, str]]] = {}
    bucket = Counter()
    n_zero = n_quar = n_frontier_dropped = 0
    quarantine = []
    for pid, sams in samples.items():
        if pid in leaked or pid not in gold:
            continue
        g = gold[pid]
        correct = [s for s in sams if strict_answer(s) == g and is_trainable(s)[0]]
        if not correct:
            loose = [s for s in sams if extract_answer(s) == g and not HANGUL.search(s)]
            if loose:
                n_quar += 1
                quarantine.append({"id": pid, "gold": g, "solutions": loose[:2]})
            n_zero += 1
            continue
        rate = len(correct) / len(sams)
        b = "easy" if rate >= 0.6 else ("mid" if rate >= 0.25 else "frontier")
        # frontier는 우연 일치 위험이 크다 → 독립 시드에서 재현되지 않으면 버린다
        if b == "frontier" and recheck:
            if not any(strict_answer(s) == g for s in recheck.get(pid, [])):
                n_frontier_dropped += 1
                continue
        n = min(pick_count(rate), len(correct))
        bucket[b] += n
        for sol in rng.sample(correct, n):
            pool.setdefault(source[pid], []).append((question[pid], sol))

    total = sum(len(v) for v in pool.values())
    print(f"\n채택 {total:,}건  버킷 {dict(bucket)}")
    print(f"  정답 0인 문항 {n_zero:,} (그중 관대 파서로만 정답이라 격리 {n_quar:,})")
    if recheck:
        print(f"  frontier 재확인 탈락 {n_frontier_dropped:,}문항 — 우연 일치로 판단")

    if args.max_records and total > args.max_records:
        keep = args.max_records / total
        print(f"\n상한 {args.max_records:,} → 출처 비례 층화 (비율 {keep:.3f})")
        for s in pool:
            k = max(1, round(len(pool[s]) * keep))
            pool[s] = rng.sample(pool[s], min(k, len(pool[s])))

    recs = [r for v in pool.values() for r in v]
    print(f"\n출처별 구성:")
    for s, v in sorted(pool.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s:<14}{len(v):>8,}  ({len(v)/len(recs)*100:5.1f}%)")

    rng.shuffle(recs)  # 학습 샘플러도 셔플하지만 파일 자체도 섞어둔다
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q, sol in recs:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": sol},
            ]}, ensure_ascii=False) + "\n")
    if quarantine:
        qp = out.with_name(out.stem + "_quarantine.jsonl")
        with qp.open("w") as f:
            for q in quarantine:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"격리 저장(폐기 아님): {qp}")
    lens = [len(s) for _, s in recs]
    lens.sort()
    print(f"\nwrote {out}  ({len(recs):,}건)")
    print(f"해설 길이 중앙 {lens[len(lens)//2]:,}자  (v2 1,173자 — 1,800자 이상이면 체제 이동)")


if __name__ == "__main__":
    main()

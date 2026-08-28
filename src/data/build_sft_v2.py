"""sft_v2 구축 (서버 실행 가능 — STaR 생성물 + 보충 풀 결합).

입력:
  - STaR 8샘플: experiments 또는 /workspace 의 train_samp8.jsonl
  - 한계선 16샘플: hard_samp16.jsonl (있으면 병합 — 문항당 최대 24샘플)
  - data/processed/train_split.csv (정답)
  - data/processed/supplement_pool.csv (전멸 문항 외부 해설)

채택 규칙 (해결률 역비례 — DART-Math 방식, 복제 대신 다양성):
  - 해결률 >= 0.6      : 정답 풀이 1개 (무작위)
  - 0.25 <= 해결률 < 0.6: 2개
  - 0 < 해결률 < 0.25  : 최대 4개
  - 0 (전멸)           : 보충 풀 해설 1개 (있는 경우)
  * 무작위 선택 (최단 선택 금지 — 압축 편향 방지, exp-003a 교훈)

사용: python src/data/build_sft_v2.py --gens <dir> --output data/processed/sft_v2.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.parse import extract_answer  # noqa: E402
from src.inference.generate import SYSTEM_PROMPT  # noqa: E402


def pick_count(rate: float) -> int:
    if rate >= 0.6:
        return 1
    if rate >= 0.25:
        return 2
    return 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="train_samp8.jsonl / hard_samp16.jsonl 이 있는 디렉토리")
    ap.add_argument("--output", default=str(ROOT / "data/processed/sft_v2.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tr = pd.read_csv(ROOT / "data/processed/train_split.csv")
    gold = dict(zip(tr.id, tr.answer))
    question = dict(zip(tr.id, tr.question))

    samples: dict[str, list[str]] = {}
    gdir = Path(args.gens)
    for fname in ["train_samp8.jsonl", "hard_samp16.jsonl"]:
        p = gdir / fname
        if not p.exists():
            print(f"[skip] {p} 없음")
            continue
        for line in p.open():
            r = json.loads(line)
            samples.setdefault(r["id"], []).extend(r["samples"])
    print(f"샘플 보유 문항: {len(samples):,}")

    sup = {}
    sup_path = ROOT / "data/processed/supplement_pool.csv"
    if sup_path.exists():
        sdf = pd.read_csv(sup_path)
        sup = dict(zip(sdf.id, sdf.solution))
        print(f"보충 풀: {len(sup):,}")

    recs = []
    stats = {"star": 0, "supplement": 0, "uncovered": 0}
    per_bucket = {}
    for pid, sams in samples.items():
        g = gold[pid]
        correct = [s for s in sams if extract_answer(s) == g]
        rate = len(correct) / len(sams)
        if correct:
            n = min(pick_count(rate), len(correct))
            chosen = rng.sample(correct, n)
            bucket = "easy" if rate >= 0.6 else ("mid" if rate >= 0.25 else "frontier")
            per_bucket[bucket] = per_bucket.get(bucket, 0) + n
            stats["star"] += n
        elif pid in sup:
            chosen = [sup[pid]]
            stats["supplement"] += 1
        else:
            stats["uncovered"] += 1
            continue
        for sol in chosen:
            recs.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question[pid]},
                        {"role": "assistant", "content": sol},
                    ]
                }
            )

    rng.shuffle(recs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsft_v2: {len(recs):,}건 -> {out}")
    print(f"  STaR 채택 {stats['star']:,} (버킷별 {per_bucket}) / 보충 {stats['supplement']:,} / 미커버 문항 {stats['uncovered']:,}")
    lens = pd.Series([len(r["messages"][2]["content"]) for r in recs])
    print(f"  해설 길이: 중앙값 {lens.median():.0f}자 / p90 {lens.quantile(0.9):.0f}자")


if __name__ == "__main__":
    main()

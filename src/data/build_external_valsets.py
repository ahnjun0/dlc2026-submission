"""외부 검증셋 구축: AIMO 우승팀(AIMO1)이 쓴 검증셋 구성을 재현한다.

- AIME (AI-MO/aimo-validation-aime, 90문항)
- AMC  (AI-MO/aimo-validation-amc, 83문항)
- MATH level 4/5 (AI-MO/aimo-validation-math-level-4/5)

각 셋에서 정답이 정수인 문항만 남긴다 (대회 metric 정렬).
출력: data/external/valsets/<name>.csv (id,question,answer)

사용: python src/data/build_external_valsets.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.parse import normalize_to_int  # noqa: E402

SOURCES = [
    # (출력명, HF repo, 문제 컬럼 후보, 정답 컬럼 후보)
    ("aime", "AI-MO/aimo-validation-aime", ["problem", "question"], ["answer"]),
    ("amc", "AI-MO/aimo-validation-amc", ["problem", "question"], ["answer"]),
    ("math_l4", "AI-MO/aimo-validation-math-level-4", ["problem", "question"], ["answer"]),
    ("math_l5", "AI-MO/aimo-validation-math-level-5", ["problem", "question"], ["answer"]),
]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {candidates} in {list(df.columns)}")


def main() -> None:
    from datasets import load_dataset

    out_dir = ROOT / "data/external/valsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, repo, q_cands, a_cands in SOURCES:
        try:
            ds = load_dataset(repo, split="train")
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {name}: {repo} 로드 실패 — {e}")
            continue
        df = ds.to_pandas()
        assert isinstance(df, pd.DataFrame)
        q_col, a_col = pick_col(df, q_cands), pick_col(df, a_cands)

        rows = []
        for i, (q, a) in enumerate(zip(df[q_col], df[a_col])):
            v = normalize_to_int(str(a))
            if v is not None:
                rows.append({"id": f"{name}-{i:04d}", "question": q, "answer": v})
        out = pd.DataFrame(rows)
        out.to_csv(out_dir / f"{name}.csv", index=False)
        summary.append((name, len(df), len(out)))
        print(f"[OK] {name}: {len(df)}문항 중 정수답 {len(out)}개 -> {name}.csv")

    print("\n요약:")
    for name, total, kept in summary:
        print(f"  {name}: {kept}/{total}")


if __name__ == "__main__":
    main()

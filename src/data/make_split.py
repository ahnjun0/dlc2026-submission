"""train.csv에서 층화 holdout 검증셋을 분리한다.

층화 기준: LaTeX 포함 여부 × 답 크기 버킷 (leaderboard와 분포가 거의 같음이
확인되었으므로, 이 두 축만 맞추면 val이 LB를 잘 대변한다).

사용:
    python src/data/make_split.py [--val-size 500] [--seed 42]

출력:
    data/processed/train_split.csv  (학습용)
    data/processed/val_split.csv    (로컬 검증용 — 학습에 절대 사용 금지)
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def answer_bucket(a: int) -> str:
    if a < 0:
        return "neg"
    if a < 10:
        return "0-9"
    if a < 1000:
        return "10-999"
    if a < 10**6:
        return "1e3-1e6"
    return "big"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data/raw/deep_chal_math_train.csv")
    df["has_latex"] = df.question.str.contains(r"[$\\]", regex=True)
    df["bucket"] = df.answer.map(answer_bucket)
    df["stratum"] = df.has_latex.astype(str) + "|" + df.bucket

    frac = args.val_size / len(df)
    val = (
        df.groupby("stratum", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=args.seed), include_groups=False)
    )
    # groupby.apply(include_groups=False)가 stratum 컬럼을 떨어뜨리므로 인덱스로 복원
    val = df.loc[val.index]
    train = df.drop(val.index)

    out = ROOT / "data/processed"
    cols = ["id", "question", "answer"]
    train[cols].to_csv(out / "train_split.csv", index=False)
    val[cols].to_csv(out / "val_split.csv", index=False)

    print(f"train {len(train)} / val {len(val)}")
    print("val 층화 분포:")
    print(val.stratum.value_counts().to_string())
    print("전체 대비 비율 차이 (%p):")
    diff = (val.stratum.value_counts(normalize=True) - df.stratum.value_counts(normalize=True)) * 100
    print(diff.round(2).to_string())


if __name__ == "__main__":
    main()

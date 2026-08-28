"""경시 계열 **정제 풀** 구축 — NuminaMath-1.5의 내장 품질 플래그를 처음으로 활용한다.

배경 (2026-08-10): 지금까지 우리는 `answer`가 정수인지만 봤다. 그런데 이 데이터셋에는
품질 메타데이터가 들어 있고, 우리는 그걸 한 번도 쓰지 않았다:
  · problem_is_valid : Yes / Incomplete / More than one problem / Not a problem
  · solution_is_valid: Yes / Incomplete / **Problem not solved** / Not matched with problem
  · question_type    : math-word-problem / **MCQ** / proof / other
  · synthetic        : True(31만) / False

특히 **MCQ 146,449건**은 우리 위생 검사에서 "객관식 잔재"로 잡히던 바로 그 오염이고,
**solution_is_valid='Problem not solved' 13,446건**은 해설이 문제를 못 푼 항목이다.

정수 필터도 정정: 콤마를 무조건 지우면 **복수 답 "1,2,3"이 123으로 둔갑**한다
(실측 1.1%, 경시 슬라이스에 편중). 천단위 구분만 허용한다.

사용:
  python src/data/build_contest_pool.py --output data/processed/contest_pool.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONTEST = ["olympiads", "aops_forum", "cn_contest", "amc_aime"]

# 문제 자리에 **해설문**이 들어간 항목 (2026-08-10 실측 197건, 0.50%).
# `problem_is_valid=Yes`를 통과하므로 플래그로는 안 걸린다. 이런 항목은 문제 안에
# 답을 향한 논증이 통째로 들어 있어서, 학습시키면 우리가 4연패한 **"주입형"**이 된다.
SOLUTION_LEAD = re.compile(
    r"^\s*(since|because|note that|we (have|know|can|see)|let's|first,"
    r"|by (the )?(am-gm|cauchy|symmetry)|observe that|assume)", re.I)
# 외부 이미지·그림 참조 (대회 train에서 127건 잡았던 것과 같은 유형)
IMAGE_REF = re.compile(r"https?://|\[img\]|\\includegraphics|(see|as shown in) (the )?figure", re.I)


def gold_int(a) -> int | None:
    """gold 답이 **단일 정수**일 때만 값을 준다.

    **콤마 표기는 전부 거부한다** (2026-08-11 실측). 종전에는 천단위 구분
    `-?\\d{1,3}(,\\d{3})+`을 허용했는데, **각 원소가 3자리인 복수 답과 형태가 같다**:
      '145,150,295' / '225,256,361' / '60,180,220,340'  ← 전부 답이 여러 개인 문항
    경시 4종에서 콤마 형태 gold 359건 중 그룹 3개 이상이 217건으로, 대종이 복수 답이었다.
    2그룹('12,100')도 천단위인지 두 답인지 구분이 불가능하다.
    잃는 것은 진짜 천단위 표기(예: '1,400,000') 소수뿐이고, 그 대가로 **오염을 원천 차단**한다.
    """
    s = str(a).strip().replace("\u2212", "-")
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=CONTEST)
    ap.add_argument("--output", default="data/processed/contest_pool.csv")
    ap.add_argument("--max-per-slice", type=int, default=0, help="0=전량")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    df = pd.DataFrame({
        "question": ds["problem"], "answer": ds["answer"], "source": ds["source"],
        "pv": ds["problem_is_valid"], "sv": ds["solution_is_valid"],
        "qt": ds["question_type"], "syn": ds["synthetic"],
    })
    df = df[df.source.isin(args.slices)]
    n0 = len(df)
    print(f"경시 4종 원본: {n0:,}\n단계별 잔존:")

    steps = [
        ("problem_is_valid=Yes", lambda d: d[d.pv == "Yes"]),
        ("solution_is_valid=Yes", lambda d: d[d.sv == "Yes"]),
        ("question_type=math-word-problem (MCQ·proof 제외)", lambda d: d[d.qt == "math-word-problem"]),
        ("gold=단일 정수 (복수답 거부)", lambda d: d[d.answer.map(gold_int).notna()]),
    ]
    for name, fn in steps:
        df = fn(df)
        print(f"  {name:<48}{len(df):>9,}  ({len(df)/n0*100:5.1f}%)")

    # --- 규칙 준수 dedup: LB 831 / 대회 train / 우리 val / 기존 외부 풀 ---
    banned: set[str] = set()
    for path, col in [
        ("data/raw/deep_chal_math_leaderboard_filtered.csv", "question"),
        ("data/raw/deep_chal_math_train.csv", "question"),
        ("data/processed/val_split_corrected.csv", "question"),
        ("data/processed/external_star_pool.csv", "question"),
    ]:
        p = ROOT / path
        assert p.exists(), f"dedup 대상 누락: {path} — 규칙 6 위반 위험, 중단"
        banned |= set(pd.read_csv(p)[col].map(norm))
    df["norm"] = df.question.map(norm)
    before = len(df)
    df = df[~df.norm.isin(banned)].drop_duplicates("norm")
    print(f"  {'dedup (LB·train·val·기존풀) + 자체 중복':<48}{len(df):>9,}  ({len(df)/n0*100:5.1f}%)")
    print(f"    └ 제거 {before - len(df):,}건")

    for name, pat in [("해설문이 문제 자리에", SOLUTION_LEAD), ("이미지·그림 참조", IMAGE_REF)]:
        hit = df.question.map(lambda q: bool(pat.search(str(q))))
        print(f"  {name + ' 제거':<48}{(~hit).sum():>9,}  (−{int(hit.sum()):,})")
        df = df[~hit]

    if args.max_per_slice:
        df = df.groupby("source", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.max_per_slice), random_state=args.seed))

    df = df.reset_index(drop=True)
    out = pd.DataFrame({
        "id": [f"cont-{i:06d}" for i in range(len(df))],
        "question": df.question.values,
        "answer": [gold_int(a) for a in df.answer.values],
        "source": df.source.values,
        "synthetic": df.syn.values,
    })
    assert out.id.is_unique and out.answer.notna().all()
    assert not set(out.question.map(norm)) & banned, "dedup 누출"
    print(f"\n슬라이스별:\n{out.source.value_counts().to_string()}")
    print(f"synthetic 비율: {out.synthetic.mean()*100:.1f}%")
    dest = ROOT / args.output
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(out):,}문항)")


if __name__ == "__main__":
    main()

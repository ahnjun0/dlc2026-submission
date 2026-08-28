"""대회 train 문항이 어느 공개 데이터셋에서 왔는지 식별한다.

목적: SFT 데이터의 도메인 매칭 (분포 정렬). leaderboard/test 문항은 분석에서 제외 —
테스트 정답을 외부에서 찾는 행위는 부정행위이며, 이 스크립트는 train만 사용한다.

매칭: 질문 텍스트 정규화(소문자, 공백 축약) 후 완전 일치 + 앞 150자 접두 일치.

사용: python src/data/match_train_sources.py [--datasets gsm8k math_qa orca_math metamath]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# repo, split, question 컬럼(또는 추출 함수)
CANDIDATES = {
    "gsm8k": ("openai/gsm8k", "main", "train", "question"),
    "math_qa": ("allenai/math_qa", None, "train", "Problem"),
    "orca_math": ("microsoft/orca-math-word-problems-200k", None, "train", "question"),
    "metamath": ("meta-math/MetaMathQA", None, "train", "query"),
    "numina15": ("AI-MO/NuminaMath-1.5", None, "train", "problem"),
    # 미매칭 12.3% 후보군 (Sonnet 웹 조사 2026-08-01 기반)
    "gsm_plus": ("qintongli/GSM-Plus", None, "test", "question"),
    "omni_math": ("KbsdJames/Omni-MATH", None, "test", "problem"),
    "svamp": ("ChilleD/SVAMP", None, "train", "Body"),
    "asdiv": ("EleutherAI/asdiv", None, "validation", "body"),
}


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math_qa", "orca_math", "metamath"])
    args = ap.parse_args()

    from datasets import load_dataset

    train = pd.read_csv(ROOT / "data/raw/deep_chal_math_train.csv")
    train["norm"] = train.question.map(norm)
    train["prefix"] = train.norm.str[:150]
    exact_index = dict(zip(train.norm, train.id))
    prefix_index = dict(zip(train.prefix, train.id))
    print(f"train: {len(train)}문항\n")

    hits_by_id: dict[str, list[str]] = {}
    for name in args.datasets:
        repo, config, split, qcol = CANDIDATES[name]
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {name}: {e}")
            continue

        # numina15는 내부 source 필드(olympiads/amc_aime/cn_k12 등)까지 기록
        sub_sources = ds["source"] if name == "numina15" and "source" in ds.column_names else None
        sub_hits: dict[str, int] = {}
        exact_ids, prefix_ids = set(), set()
        for i, q in enumerate(ds[qcol]):
            n = norm(q)
            tid = exact_index.get(n) or prefix_index.get(n[:150])
            if tid:
                (exact_ids if n in exact_index else prefix_ids).add(tid)
                if sub_sources is not None:
                    sub_hits[sub_sources[i]] = sub_hits.get(sub_sources[i], 0) + 1
        for tid in exact_ids | prefix_ids:
            hits_by_id.setdefault(tid, []).append(name)
        print(
            f"[{name}] 소스 {len(ds):,}건 -> train 커버: 완전일치 {len(exact_ids)}, "
            f"접두일치(추가) {len(prefix_ids - exact_ids)} "
            f"(계 {len(exact_ids | prefix_ids)}/{len(train)} = {len(exact_ids | prefix_ids)/len(train):.1%})"
        )
        if sub_hits:
            print("  내부 출처별 매칭:", dict(sorted(sub_hits.items(), key=lambda x: -x[1])))

    covered = set(hits_by_id)
    print(f"\n총 커버: {len(covered)}/{len(train)} = {len(covered)/len(train):.1%}")
    out = train[["id"]].copy()
    out["sources"] = out.id.map(lambda i: ",".join(hits_by_id.get(str(i), [])) or "unmatched")
    out.to_csv(ROOT / "data/processed/train_source_map.csv", index=False)
    print("분포:")
    print(out.sources.value_counts().head(15).to_string())

    # 미매칭 샘플 확인용
    unmatched = train[~train.id.isin(covered)]
    print("\n미매칭 예시 5건:")
    for q in unmatched.question.head(5):
        print(" -", q[:100].replace("\n", " "))


if __name__ == "__main__":
    main()

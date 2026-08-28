"""`external_star_pool.csv`(exp-024에 쓴 외부 25k)의 **출처를 사후 특정**한다.

왜 필요한가: 이 풀은 2026-08-01에 CSV만 커밋됐고 **빌드 스크립트가 남지 않았다.**
id가 `ext-NNNNNN`으로 익명화돼 원본 추적이 불가능한 상태다. 최종 제출 시 외부
데이터 목록 명시가 의무이고, 수상 시 재현 검증도 있으므로 출처 불명은 허용되지 않는다.

방법: 질문 텍스트를 정규화해 후보 데이터셋과 완전 일치·접두(150자) 일치로 대조.
`match_train_sources.py`와 동일한 정규화를 쓴다 (결과 비교 가능성 유지).

사용: python src/data/identify_ext_pool.py [--datasets numina15 openr1 omni_math ...]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CANDIDATES = {
    "numina15": ("AI-MO/NuminaMath-1.5", None, "train", "problem"),
    "openr1": ("open-r1/OpenR1-Math-220k", None, "train", "problem"),
    "omni_math": ("KbsdJames/Omni-MATH", None, "test", "problem"),
    "orca_math": ("microsoft/orca-math-word-problems-200k", None, "train", "question"),
    "metamath": ("meta-math/MetaMathQA", None, "train", "query"),
    "gsm_plus": ("qintongli/GSM-Plus", None, "test", "question"),
    "math_qa": ("allenai/math_qa", None, "train", "Problem"),
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/processed/external_star_pool.csv")
    ap.add_argument("--datasets", nargs="+", default=["numina15", "openr1", "omni_math"])
    ap.add_argument("--output", default="data/processed/ext_pool_sources.csv")
    args = ap.parse_args()

    from datasets import load_dataset

    pool = pd.read_csv(ROOT / args.pool)
    pool["norm"] = pool.question.map(norm)
    pool["prefix"] = pool.norm.str[:150]
    exact = dict(zip(pool.norm, pool.id))
    prefix = dict(zip(pool.prefix, pool.id))
    print(f"풀: {len(pool):,}문항\n")

    hits: dict[str, list[str]] = {}
    for name in args.datasets:
        repo, config, split, qcol = CANDIDATES[name]
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {name}: {e}")
            continue
        # 내부 출처 필드가 있으면(numina15의 source 등) 함께 집계
        subs = ds["source"] if "source" in ds.column_names else None
        sub_hits: dict[str, int] = {}
        found = set()
        for i, q in enumerate(ds[qcol]):
            n = norm(q)
            pid = exact.get(n) or prefix.get(n[:150])
            if pid:
                found.add(pid)
                hits.setdefault(pid, []).append(name)
                if subs is not None:
                    s = str(subs[i])
                    sub_hits[s] = sub_hits.get(s, 0) + 1
        print(f"{name:<12} 매칭 {len(found):>6,}문항  ({len(found)/len(pool)*100:.1f}%)")
        if sub_hits:
            top = sorted(sub_hits.items(), key=lambda kv: -kv[1])[:12]
            print("             내부 출처: " + " / ".join(f"{k} {v:,}" for k, v in top))

    pool["sources"] = pool.id.map(lambda i: "|".join(sorted(set(hits.get(i, [])))) or "")
    n_hit = (pool.sources != "").sum()
    print(f"\n**출처 확인 {n_hit:,} / {len(pool):,} ({n_hit/len(pool)*100:.1f}%)**")
    print(pool.sources.value_counts().head(10).to_string())
    out = ROOT / args.output
    pool[["id", "sources"]].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

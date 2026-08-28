"""문샷(long-CoT full FT) 학습 데이터 구축 — OpenR1-Math-220k.

절대 규칙 준수:
  - 정수 답 문항만 (대회 채점 형식 정렬)
  - leaderboard 831문항과 정규화 일치 시 기계적 제거 (규칙 6 dedup)
  - DATA_SOURCES.md 기록 의무 (출처: open-r1/OpenR1-Math-220k, Apache-2.0)

선택 규칙:
  - math_verify 검증 통과 + reasoning_complete 인 R1 생성물 중 첫 번째
  - 길이 상한 --max-chars (기본 24000자 ≈ seq 8192) — 초과분은 절단이 아니라 제외

사용: python src/data/build_moonshot_data.py --output data/processed/moonshot_r1.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inference.generate import SYSTEM_PROMPT  # noqa: E402


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def to_int(ans: str):
    """**단일 정수만** 인정. 콤마 표기는 전부 거부한다.

    천단위 구분을 허용하면 **각 원소가 3자리인 복수 답**('145,150,295')이 통과한다
    (2026-08-11 실측: 경시 4종 콤마 gold 359건 중 217건이 3그룹 이상 = 복수 답).
    """
    s = str(ans).strip().replace("\u2212", "-")
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "data/processed/moonshot_r1.jsonl"))
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    lb = pd.read_csv(ROOT / "data/raw/deep_chal_math_leaderboard_filtered.csv")
    lb.columns = [c.strip() for c in lb.columns]
    lb_norm = {norm_text(q) for q in lb.question}

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    stats = {"total": 0, "non_integer": 0, "no_verified_gen": 0, "too_long": 0, "lb_dup": 0, "kept": 0}
    lens = []
    with open(args.output, "w") as f:
        for r in ds:
            stats["total"] += 1
            v = to_int(r["answer"])
            if v is None:
                stats["non_integer"] += 1
                continue
            if norm_text(r["problem"]) in lb_norm:
                stats["lb_dup"] += 1
                continue
            gen = None
            oks = r["correctness_math_verify"] or []
            comp = r["is_reasoning_complete"] or []
            for g, ok, c in zip(r["generations"], oks, comp):
                if ok and c:
                    gen = g
                    break
            if gen is None:
                stats["no_verified_gen"] += 1
                continue
            if len(gen) > args.max_chars:
                stats["too_long"] += 1
                continue
            lens.append(len(gen))
            stats["kept"] += 1
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["problem"]},
                {"role": "assistant", "content": gen},
            ]}, ensure_ascii=False) + "\n")

    print(f"moonshot_r1: {stats['kept']:,}건 -> {args.output}")
    print(f"  필터: 총 {stats['total']:,} / 비정수답 {stats['non_integer']:,} / 검증생성물없음 {stats['no_verified_gen']:,} / 초과길이 {stats['too_long']:,} / LB중복 {stats['lb_dup']:,}")
    s = pd.Series(lens)
    print(f"  생성물 길이: 중앙값 {s.median():.0f} / p90 {s.quantile(.9):.0f} / 최대 {s.max():.0f}자")


if __name__ == "__main__":
    main()

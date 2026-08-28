"""**exp-120 중간 길이(concise) CoT 데이터** — 비어 있는 생성 체제를 겨냥 (2026-08-26).

우리 함대는 **두 점**만 차지한다: v2 계열 **1,412~1,441자** · 문샷 계열 **10,425~11,033자**.
그 사이 **7.3배 도약**이 비어 있다.

**규칙 대조** (착수 전)
```
5.2  공개 데이터 자유 · 모든 참가자가 무료·동등 접근 → SmallThoughts **Apache-2.0** ✅
5.2c 최종 제출 시 목록 명시 의무 → `data/external/DATA_SOURCES.md` 기록
4.1  베이스 모델 유지 · 가중치 병합 아님 ✅
4.2  SFT 는 열거된 허용 기법 ✅
6번  leaderboard/test 문항과의 중복 **기계적 제거** 필수
```

**필터** (각 단계의 이유를 적는다)
```
① 수학·정수답        우리 과제가 정수 EM 이다. `\boxed{}` 안이 정수인 것만
② **코드/TIR 제외**   추론 시 코드 실행이 금지(4.3)라 코드를 쓰고 결과를 지어내는 모델이 된다
                     (`코드형 풀이 체제` 축에서 API 상한 35.8~53.3% 로 이미 폐쇄)
③ 길이 구간          **1,800~6,000자** — 목표 체제. 너무 짧으면 v2 와 같고 길면 문샷과 같다
④ 중복 제거          대회 train ∪ leaderboard 와 정규화 대조
⑤ **시스템 프롬프트 교체**  우리 것(184자)으로 바꾼다. 학습·추론 불일치는 `exp-010` 을 죽인 원인이다
```
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.parse import extract_answer  # noqa: E402
from src.inference.generate import SYSTEM_PROMPT as SYS  # noqa: E402

CODE = re.compile(r"```|\bimport \w|\bdef \w+\(|print\(|sympy|numpy|python", re.I)
BOX = re.compile(r"\\boxed\s*\{")


def boxed_int(text: str):
    """**마지막 `\boxed{}` 안이 순수 정수일 때만** 그 값을 준다.

    `extract_answer` 는 제출 파서라 관대하다 — 복소수·분수·증명에서도 숫자를 긁어온다.
    학습 데이터에서는 그게 치명적이다: **정수를 안 내는 모델**이 만들어진다.
    실측(2026-08-26): 관대한 필터로는 순수 정수가 **29%** 뿐이고 나머지 71%가
    기호·수식·분수·부등식이었다. 중괄호 균형을 맞춰 내용을 뽑고 정수만 받는다.
    """
    m = None
    for m in BOX.finditer(text):
        pass
    if not m:
        return None
    i, depth, out = m.end(), 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        out.append(c); i += 1
    b = "".join(out).strip().replace(",", "")
    return int(b) if re.fullmatch(r"-?\d{1,18}", b) else None


def norm(q: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", str(q)).strip().lower().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--out", default="data/processed/sft_concise.jsonl")
    ap.add_argument("--min-len", type=int, default=1800)
    ap.add_argument("--max-len", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=25000)
    a = ap.parse_args()

    ban = set()
    for f, col in (("data/processed/train_split.csv", "question"),
                   ("data/processed/val_split_corrected.csv", "question"),
                   ("data/raw/deep_chal_math_leaderboard_filtered.csv", "question")):
        try:
            ban |= set(pd.read_csv(f)[col].map(norm))
        except Exception as e:
            print(f"  ⚠ 금지 목록 로드 실패 {f}: {e}")
    print(f"중복 제거 대상 {len(ban):,}문항 (대회 train ∪ val ∪ leaderboard)\n")

    rows, st = [], dict(총=0, 비정수=0, 코드=0, 길이=0, 중복=0, 채택=0)
    for p in a.parquets:
        d = pd.read_parquet(p)
        for r in d.itertuples():
            st["총"] += 1
            msgs = list(r.messages)
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if not user or not asst:
                continue
            # ① **엄격한 정수 답** — boxed 안이 순수 정수여야 한다
            v = boxed_int(asst)
            if v is None:
                st["비정수"] += 1; continue
            # ② 코드/TIR
            if CODE.search(asst):
                st["코드"] += 1; continue
            # ③ 길이 구간
            if not (a.min_len <= len(asst) <= a.max_len):
                st["길이"] += 1; continue
            # ④ 중복
            q = re.sub(r"^Return your final response within \\boxed\{\}\.\s*", "", user).strip()
            if norm(q) in ban:
                st["중복"] += 1; continue
            st["채택"] += 1
            # ⑤ **우리 시스템 프롬프트**로 교체
            rows.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": q},
                {"role": "assistant", "content": asst},
            ]})
            if len(rows) >= a.limit:
                break
        if len(rows) >= a.limit:
            break

    print("%-10s %10s" % ("단계", "건수"))
    for k, v in st.items():
        print("%-10s %10s" % (k, f"{v:,}"))
    L = pd.Series([len(r["messages"][2]["content"]) for r in rows])
    print(f"\n채택 {len(rows):,}건 · assistant 길이 중앙 **{L.median():,.0f}자** "
          f"(p10 {L.quantile(.1):,.0f} · p90 {L.quantile(.9):,.0f})")
    print(f"  참고: v2 생성 1,412자 · 문샷 10,425자 → **목표 구간에 있는가**")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()

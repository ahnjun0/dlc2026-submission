"""**exp-101 적응형 캐스케이드** 제출 CSV 생성기.

**규칙** (val460 에서 +0.43%p · P 98.8%, 3시드·교차적합 5분할 전부 양수):
```
1. v2 32표의 최다 득표가 --confirm(기본 30) 이상  →  **v2 단독으로 확정** (문샷 생략)
2. 아니면  v2 + 문샷 풀링 (64표)
3. 그 결과의 1·2위 격차가 --margin(기본 5) 이하  →  **경시STaR 32표 증원** (96표)
```
**gold 도 출처도 안 본다** — 표 구조만으로 분기하므로 8/4 유권해석의
*"유형별 특화 + 라우팅"* 에 해당하지 않는다.

주 모델(동률 우선권)은 항상 v2 다 — `pooled_vote` 계약상 첫 인자.

사용:
  python src/eval/build_cascade.py --v2 A.jsonl --partner B.jsonl --third C.jsonl \
      --input data/raw/leaderboard.csv --out submissions/sub-0NN_cascade.csv
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src.eval.parse import extract_answer  # noqa: E402
from src.eval.score import pooled_vote  # noqa: E402


def load(path, expect_n=32, strict=True, allowed_n=None):
    """생성물 로더.

    **중복 id 를 조용히 덮어쓰지 않는다** — resume 이 중간에 겹치면 같은 문항이 두 번
    기록될 수 있고, 나중 것이 부분 생성물이면 표가 줄어든 채로 판정에 들어간다.
    **표 수도 검증한다** — 32표를 기대하는데 8표만 있으면 그 문항만 조용히 약해진다.
    (2026-08-25 외부 검토 지적)
    """
    # allowed_n: 허용되는 표 수의 집합. 2단계 조기 종료를 쓰면 문샷이 16 또는 32 다
    #            (`docs/confirm-threshold-2026-08-27.md`). **중복 id 검사는 그대로 유지한다.**
    ok_n = set(allowed_n) if allowed_n else ({expect_n} if expect_n else set())
    out, dup, short = {}, [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            i = r["id"]
            if i in out:
                dup.append(i); continue          # 첫 기록을 유지한다
            v = [extract_answer(s) for s in r["samples"]]
            if ok_n and len(v) not in ok_n:
                short.append((i, len(v)))
            out[i] = v
    if dup or short:
        msg = []
        if dup:
            msg.append(f"중복 id {len(dup)}건 (예: {dup[:3]})")
        if short:
            msg.append(f"표 수가 {sorted(ok_n)} 중 하나가 아닌 문항 {len(short)}건 (예: {short[:3]})")
        head = f"{path}: " + " · ".join(msg)
        # **기본값은 중단이다** — resume 이 겹쳐 부분 레코드가 먼저 기록되면
        # 잘못된 표로 제출물이 만들어진다. 경고만 하고 진행하면 아무도 못 본다.
        # 복구 목적일 때만 --allow-partial-gens 로 명시적으로 진행한다 (2026-08-25 외부 지적).
        if strict:
            raise SystemExit(f"**중단** — {head}\n"
                             f"  생성을 --resume 으로 마저 돌리거나, 의도한 것이면 "
                             f"--allow-partial-gens 를 명시할 것")
        print(f"⚠ {head} — --allow-partial-gens 로 진행함")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True, help="주 모델 생성물 (동률 우선권)")
    ap.add_argument("--partner", required=True, help="2단계 파트너 (문샷)")
    ap.add_argument("--third", required=True, help="3단계 증원 (경시STaR)")
    ap.add_argument("--input", required=True, help="문항 CSV (id 검증용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm", type=int, default=30, help="v2 최다 득표 이 값 이상이면 확정")
    # **1단계 2단 분할** (2026-08-26 신설 · 기본 off 로 기존 동작 보존)
    # 앞 N표만으로 강하게 합의하면 나머지를 안 뽑는다. vLLM 은 자식 시드를 seed…seed+n-1 로
    # 만들므로 **n=32 생성물의 앞 16개 = n=16 실행 결과**다 → 시뮬레이션이 타당하다.
    # val 3시드에서 답이 **완전히 동일**했고(시드별 [387,383,385] 불변) v2 생성량만 65.5% 로 준다.
    ap.add_argument("--tier1-n", type=int, default=0,
                    help="1단계 조기 확정에 쓸 앞 표 수 (0=사용 안 함)")
    ap.add_argument("--tier1-confirm", type=int, default=0,
                    help="앞 --tier1-n 표 중 최다 득표가 이 값 이상이면 즉시 확정")
    ap.add_argument("--margin", type=int, default=5, help="1·2위 격차 이 값 이하면 3단계 증원")
    ap.add_argument("--partner-early-stop", action="store_true",
                    help="문샷이 문항별로 16 또는 32표를 가질 수 있게 허용 (2단계 조기 종료). "
                         "중복 id 검사는 유지된다")
    ap.add_argument("--allow-partial-gens", action="store_true",
                    help="생성물의 중복 id·표 수 불일치가 있어도 진행 (기본은 중단 — 복구 전용)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="생성물 결손을 감수하고 진행 (기본은 중단 — 조용한 축소 방지)")
    a = ap.parse_args()

    st = not a.allow_partial_gens
    v2 = load(a.v2, strict=st)
    pa = load(a.partner, strict=st, allowed_n={16, 32} if a.partner_early_stop else None)
    th = load(a.third, strict=st)
    ids = list(pd.read_csv(a.input)["id"])

    rows, n1, n2, n3 = [], 0, 0, 0
    miss2, miss3 = [], []          # **조용한 축소 감시** (아래 참조)
    n_t1 = 0
    for i in ids:
        g2 = v2.get(i, [])
        # **1단계-a: 앞 N표 조기 확정** (--tier1-n 이 있을 때만)
        if a.tier1_n and a.tier1_confirm:
            head = g2[:a.tier1_n]
            ch = Counter(x for x in head if x is not None)
            if ch and max(ch.values()) >= a.tier1_confirm:
                rows.append({"id": i, "answer": pooled_vote([head])})
                n_t1 += 1; n1 += 1
                continue
        c = Counter(x for x in g2 if x is not None)
        if c and max(c.values()) >= a.confirm:          # 1단계
            ans, n1 = pooled_vote([g2]), n1 + 1
        else:
            if i not in pa:
                miss2.append(i)
            groups = [g2, pa.get(i, [])]
            ans, n2 = pooled_vote(groups), n2 + 1
            pooled = Counter(x for g in groups for x in g if x is not None)
            top = pooled.most_common(2)
            gap = top[0][1] - top[1][1] if len(top) > 1 else top[0][1] if top else 0
            if gap <= a.margin:                          # 3단계
                if i not in th:
                    miss3.append(i)
                groups.append(th.get(i, []))
                ans, n3 = pooled_vote(groups), n3 + 1
        rows.append({"id": i, "answer": ans})

    # **단계가 중간에 죽으면 그 문항은 소리 없이 앞 단계 결과로 떨어진다.**
    # 점수는 나오지만 우리가 채택한 구성이 아니다 — 반드시 눈에 띄게 만든다.
    if miss2 or miss3:
        print(f"⚠ 생성물 결손: 2단계 {len(miss2)}문항 · 3단계 {len(miss3)}문항")
        print(f"  예: {(miss2 or miss3)[:5]}")
        if not a.allow_partial:
            raise SystemExit(
                "중단 — 해당 단계를 --resume 으로 마저 돌리거나, "
                "의도한 축소면 --allow-partial 을 명시할 것")

    df = pd.DataFrame(rows)
    # **강제 검증** — 제출 실패는 되돌릴 수 없다
    assert len(df) == len(ids), f"행 수 불일치 {len(df)} != {len(ids)}"
    assert list(df["id"]) == ids, "id 순서/내용 불일치"
    assert df["id"].duplicated().sum() == 0, "id 중복"
    nblank = int(df["answer"].isna().sum())
    # **int64 로 캐스팅하지 않는다** — 모델이 int64 상한(9.22e18)을 넘는 답을 다수로 내면
    # `astype("int64")` 가 OverflowError 로 죽는다. 확실한 오답이지만 **파이프라인이 죽으면
    # 제출 자체가 불가능**하다. `make_submission.py` 는 처음부터 문자열로 보존해 왔는데
    # 이 조립기가 그 방어를 되돌렸었다 (2026-08-25 외부 검토로 적발·재현).
    df["answer"] = df["answer"].map(lambda v: "0" if v is None else str(int(v)))
    assert list(df.columns) == ["id", "answer"], "헤더는 반드시 소문자 id,answer"
    assert df["answer"].str.fullmatch(r"-?\d+").all(), "정수 문자열이 아닌 답이 있다"

    _huge = int((df["answer"].str.len() > 19).sum())
    if _huge:
        print(f"⚠ int64 초과 답 {_huge}건 — 문자열로 보존한다 (거의 확실한 오답이나 제출은 진행)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    if a.tier1_n:
        print(f"1단계-a 조기 확정 {n_t1}/{len(ids)} ({n_t1/len(ids):.1%}) "
              f"— v2 표 사용 {(a.tier1_n*n_t1 + 32*(len(ids)-n_t1))/(32*len(ids)):.1%}")
    print(f"1단계 v2 확정 {n1} · 2단계 풀링 {n2} · 그중 3단계 증원 {n3}")
    print(f"문샷 절감 {n1/len(ids):.1%} · 경시 실행 {n3/len(ids):.1%} · 결측 {nblank}")
    print(f"→ {a.out} ({len(df)}행)")


if __name__ == "__main__":
    main()

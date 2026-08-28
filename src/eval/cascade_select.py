"""**캐스케이드 다음 단계 대상 문항을 골라 CSV 로 뽑는다** (exp-101).

D-Day 파이프라인은 단계 간 **의존**이 있다 — 2단계는 1단계 결과를 봐야 대상이 정해지고,
3단계는 2단계 결과를 봐야 한다. 이 스크립트가 그 경계를 만든다.

```
--stage 2   v2 최다 득표 < confirm 인 문항      (문샷을 돌릴 대상)
--stage 3   v2+문샷 풀링의 1·2위 격차 <= margin (경시STaR 을 돌릴 대상)
```

**출력은 원본 CSV 의 부분집합**이라 `generate.py --input` 에 그대로 넣을 수 있다.
원본의 모든 컬럼을 보존한다 (문제 본문이 있어야 생성이 된다).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.parse import extract_answer  # noqa: E402


def load(path, expect_n=32, strict=True):
    """생성물 로더.

    **중복 id 를 조용히 덮어쓰지 않는다** — resume 이 중간에 겹치면 같은 문항이 두 번
    기록될 수 있고, 나중 것이 부분 생성물이면 표가 줄어든 채로 판정에 들어간다.
    **표 수도 검증한다** — 32표를 기대하는데 8표만 있으면 그 문항만 조용히 약해진다.
    (2026-08-25 외부 검토 지적)
    """
    out, dup, short = {}, [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            i = r["id"]
            if i in out:
                dup.append(i); continue          # 첫 기록을 유지한다
            v = [extract_answer(s) for s in r["samples"]]
            if expect_n and len(v) != expect_n:
                short.append((i, len(v)))
            out[i] = v
    if dup or short:
        msg = []
        if dup:
            msg.append(f"중복 id {len(dup)}건 (예: {dup[:3]})")
        if short:
            msg.append(f"표 수가 {expect_n}이 아닌 문항 {len(short)}건 (예: {short[:3]})")
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
    ap.add_argument("--stage", type=int, choices=[2, 3], required=True)
    ap.add_argument("--input", required=True, help="원본 문항 CSV")
    ap.add_argument("--v2", required=True)
    ap.add_argument("--partner", help="stage 3 에 필요")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing-partner", action="store_true",
                    help="2단계 생성물이 없는 문항을 3단계에서 **제외**하고 진행 (fail-closed). "
                         "기본은 중단 — 조용한 축소를 막기 위해서다")
    ap.add_argument("--allow-partial-gens", action="store_true",
                    help="생성물의 중복 id·표 수 불일치가 있어도 진행 (기본은 중단 — 복구 전용)")
    ap.add_argument("--confirm", type=int, default=30)
    ap.add_argument("--margin", type=int, default=5)
    a = ap.parse_args()

    src = pd.read_csv(a.input)
    v2 = load(a.v2, strict=not a.allow_partial_gens)
    missing = [i for i in src["id"] if i not in v2]
    if missing:
        # **조용한 축소를 막는다** — 1단계가 덜 끝났는데 2단계를 돌리면
        # 그 문항들은 영영 v2 단독으로 남는다.
        sys.exit(f"오류: v2 생성물에 {len(missing)}문항이 없다 (예: {missing[:3]}). "
                 f"1단계를 --resume 으로 마저 돌릴 것.")

    if a.stage == 2:
        sel = []
        for i in src["id"]:
            c = Counter(x for x in v2[i] if x is not None)
            if not (c and max(c.values()) >= a.confirm):
                sel.append(i)
        why = f"v2 최다 < {a.confirm}"
    else:
        pa = load(a.partner, strict=not a.allow_partial_gens)
        sel, miss_partner = [], []
        for i in src["id"]:
            c = Counter(x for x in v2[i] if x is not None)
            if c and max(c.values()) >= a.confirm:
                continue                      # 1단계에서 확정됨 — 3단계 대상 아님
            if i not in pa:
                # **fail-closed**: 문샷이 없는 문항은 3단계 대상이 될 수 없다 —
                # 3단계는 «v2+문샷 풀링의 1·2위 격차» 로 판정하는데 그 풀링이 성립하지 않는다.
                # 그런 문항은 v2 단독으로 남는 것이 올바른 축소다.
                # 종전에는 **무조건 sys.exit** 이라, 2단계가 시간 상한에 걸린 D-Day 에
                # 파이프라인이 여기서 멈췄다 (2026-08-26 리허설로 적발).
                # `--allow-partial-gens` 는 로더의 중복·표수 검사만 덮으므로 별도 플래그가 필요하다.
                if a.allow_missing_partner:
                    miss_partner.append(i); continue
                sys.exit(f"오류: 문샷 생성물에 {i} 가 없다. 2단계를 마저 돌릴 것. "
                         f"(의도한 축소면 --allow-missing-partner)")
            pooled = Counter(x for g in (v2[i], pa[i]) for x in g if x is not None)
            top = pooled.most_common(2)
            gap = top[0][1] - top[1][1] if len(top) > 1 else (top[0][1] if top else 0)
            if gap <= a.margin:
                sel.append(i)
        why = f"풀링 격차 <= {a.margin}"

    out = src[src["id"].isin(set(sel))].copy()
    assert len(out) == len(sel), f"선별 손실 {len(out)} != {len(sel)}"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    if a.stage == 3 and miss_partner:
        print(f"⚠ **fail-closed**: 문샷이 없어 3단계에서 제외한 문항 {len(miss_partner)}개 "
              f"(예: {miss_partner[:3]}) — 이 문항들은 v2 단독으로 남는다")
    print(f"{a.stage}단계 대상 {len(out)}/{len(src)} ({len(out)/len(src):.1%}) — {why}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()

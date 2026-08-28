"""Exact Match 채점과 다수결(maj@k) 집계."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


def exact_match(pred: int | None, gold: int) -> bool:
    return pred is not None and pred == gold


def majority_vote(preds: Iterable[int | None]) -> int | None:
    """파싱 실패(None)를 제외한 다수결. 동률이면 먼저 등장한 답.

    단일 모델 투표용. **풀링(여러 모델 표 합산)에는 `pooled_vote`를 쓸 것** —
    이 함수는 동률을 등장 순서로 풀기 때문에 모델을 붙이는 순서가 결과를 바꾼다
    (실측: v2+문샷 64표에서 v2를 먼저 붙이면 83.53%, 문샷을 먼저면 83.32%).
    """
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    for p in valid:  # 등장 순서로 동률 해소
        if counts[p] == best:
            return p
    return None


def pooled_vote(groups: Sequence[Sequence[int | None]]) -> int | None:
    """여러 모델의 표를 합산한 다수결.

    **정확한 계약** (2026-08-07 외부 리뷰로 문언 교정):
      · 각 모델 **내부의 샘플 순서**에는 완전히 무관하다
      · 모델 **우선순위는 인자 순서로 의도적으로 고정**된다 —
        `[v2, moonshot]`과 `[moonshot, v2]`는 다를 수 있으며 이는 버그가 아니다.
        호출부는 반드시 **주 모델을 첫 인자로** 넘겨야 한다.

    groups는 **우선순위 순서**로 준다 (groups[0] = 주 모델).
    동률 해소: ① 주 모델의 지지표가 많은 답 → ② 그다음 모델 → ③ 최종적으로 작은 값.

    설계 근거 (2026-08-06 실측, val460 4시드):
      · 동률은 드물지만(64표에서 1.5%) 1문항을 좌우하고, 1위와 격차가 1문항이다
      · 등장순 정책은 **모델을 붙이는 순서에 따라 83.53% ↔ 83.32%로 흔들린다**
      · 이 정책은 양방향 모두 83.64%로 동일 (등장순 정책의 두 값보다 높음)
      · 단일 모델에 적용해도 총점이 바뀌지 않았다(v2 4시드 82.17% 불변) —
        동률 문항 자체가 2.6%로 드물기 때문. 단 개별 동률에서는 등장순과
        다를 수 있다(`[2,1]` → 기존 2, 이 함수 1)
    """
    valid = [p for g in groups for p in g if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    tied = [a for a, n in counts.items() if n == best]
    if len(tied) == 1:
        return tied[0]

    def rank(answer: int) -> tuple:
        support = tuple(-sum(1 for p in g if p == answer) for g in groups)
        return (*support, answer)  # 지지표 많은 순 → 값이 작은 순

    return min(tied, key=rank)


def accuracy(preds: Sequence[int | None], golds: Sequence[int]) -> float:
    assert len(preds) == len(golds)
    if not golds:
        return 0.0
    return sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(golds)


def maj_at_k(samples_per_problem: Sequence[Sequence[int | None]], golds: Sequence[int], k: int) -> float:
    """문항별 샘플 답 리스트에서 앞 k개로 다수결 정확도를 계산한다."""
    preds = [majority_vote(s[:k]) for s in samples_per_problem]
    return accuracy(preds, golds)

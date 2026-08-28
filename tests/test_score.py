

def test_pooled_vote_independent_of_sample_order_within_groups():
    """계약: **각 모델 내부의 샘플 순서**에는 완전히 무관해야 한다.

    (모델 우선순위는 인자 순서로 **의도적으로** 고정된다 — 아래 별도 테스트.)
    종전 테스트는 `pooled_vote([a,b]) == pooled_vote([a,b])`로 같은 식을 두 번
    비교하는 공허한 단언이었다 (2026-08-07 외부 리뷰 지적).
    """
    import itertools

    from src.eval.score import pooled_vote
    a, b = [1, 1, 2], [2, 3, 3]          # 1:2표, 2:2표, 3:2표 — 3중 동률
    expected = pooled_vote([a, b])
    for pa in itertools.permutations(a):
        for pb in itertools.permutations(b):
            assert pooled_vote([list(pa), list(pb)]) == expected


def test_pooled_vote_model_priority_is_deliberate():
    """모델 우선순위는 인자 순서로 정해지며, 바뀌면 결과가 달라질 수 있다.

    이것이 '버그'가 아니라 '계약'이다 — 주 모델이 더 강하므로 동률 시 그쪽 손을
    들어준다. 호출부는 반드시 **주 모델을 첫 인자로** 넘겨야 한다.
    """
    from src.eval.score import pooled_vote
    a, b = [1, 1, 2], [2, 3, 3]
    assert pooled_vote([a, b]) == 1      # 주 모델 a가 1을 2표 지지
    assert pooled_vote([b, a]) == 3      # 주 모델 b가 3을 2표 지지


def test_pooled_vote_matches_majority_when_no_tie():
    """동률이 없으면 기존 동작과 완전히 같다 (val460 실측 98% 이상의 문항)."""
    from src.eval.score import majority_vote, pooled_vote
    for preds in ([5, 5, 7], [1, 2, 2, 3], [None, 4, 4], [9]):
        assert pooled_vote([preds]) == majority_vote(preds)


def test_pooled_vote_may_differ_from_majority_on_ties():
    """동률일 때는 의도적으로 다르다 — 등장순(비결정) 대신 작은 값(결정론).

    val460 4시드 실측에서는 이 차이가 총점을 바꾸지 않았다(양쪽 82.17%).
    """
    from src.eval.score import majority_vote, pooled_vote
    assert majority_vote([2, 1]) == 2      # 먼저 등장한 값
    assert pooled_vote([[2, 1]]) == 1      # 결정론적으로 작은 값


def test_pooled_vote_no_tie_uses_plain_majority():
    from src.eval.score import pooled_vote
    assert pooled_vote([[7, 7, 7], [8]]) == 7


def test_pooled_vote_all_none():
    from src.eval.score import pooled_vote
    assert pooled_vote([[None, None], [None]]) is None


def test_pooled_vote_final_tiebreak_is_smallest():
    """지지표까지 완전히 같으면 작은 값 — 결정론적이어야 한다."""
    from src.eval.score import pooled_vote
    assert pooled_vote([[3, 9]]) == 3

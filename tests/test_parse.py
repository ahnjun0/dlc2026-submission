import pytest

from src.eval.parse import extract_answer, normalize_to_int
from src.eval.score import maj_at_k, majority_vote


@pytest.mark.parametrize(
    "text,expected",
    [
        # boxed 기본
        (r"So the answer is \boxed{42}.", 42),
        (r"\boxed{-17}", -17),
        (r"\boxed{ 1,234,567 }", 1234567),
        (r"\boxed{132.0}", 132),
        (r"\boxed{\frac{10}{2}}", 5),
        (r"\boxed{-\frac{10}{2}}", -5),  # -\frac 형태
        (r"\boxed{\dfrac{9}{3}}", 3),
        # 중첩 중괄호
        (r"\boxed{5^{2}}", 5),
        # 여러 boxed → 마지막 우선
        (r"\boxed{1} ... wait, actually \boxed{2}", 2),
        # 마지막 boxed가 비정수면 이전 boxed로
        (r"\boxed{7} then \boxed{unknown}", 7),
        # 결론 패턴
        ("The final answer is 650.", 650),
        ("Answer: -3", -3),
        ("따라서 정답은 5000입니다.", 5000),
        ("답: 29", 29),
        ("the answer is $12$ dollars", 12),
        # 거대 정수 (float 정밀도 손실 금지)
        (r"\boxed{3431577000000000}", 3431577000000000),
        ("The final answer is 3431577000000001.", 3431577000000001),
        # 유니코드 마이너스
        ("The final answer is −8.", -8),
        # fallback: 마지막 숫자
        ("We compute 3 + 4 = 7", 7),
        ("x = 10, so 2x = 20", 20),
        # 소수는 정수일 때만 — 비정수 소수는 잘라서 오답을 만들지 않고 실패 처리
        ("The answer is 42.5, roughly", None),
        # 완전 실패
        ("I cannot solve this.", None),
        ("", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_extract_conclusion_over_intermediate():
    text = "First, 100 + 30 = 130 minutes. Speed is 6.5. The final answer is 650."
    assert extract_answer(text) == 650


def test_normalize():
    assert normalize_to_int("1,234") == 1234
    assert normalize_to_int("-0") == 0
    assert normalize_to_int("42.000") == 42
    assert normalize_to_int("42.5") is None
    assert normalize_to_int("1e3") == 1000
    assert normalize_to_int("abc") is None


def test_majority_vote():
    assert majority_vote([1, 2, 2, None, 3]) == 2
    assert majority_vote([1, 2]) == 1  # 동률 → 먼저 등장
    assert majority_vote([None, None]) is None


def test_maj_at_k():
    samples = [[1, 1, 2, 2, 2], [5, None, 5, 4, 4]]
    golds = [2, 5]
    assert maj_at_k(samples, golds, k=2) == 0.5  # [1], [5]
    assert maj_at_k(samples, golds, k=5) == 1.0  # [2], [5 — 동률이지만 5가 먼저]


def test_huge_hallucinated_integer_does_not_crash():
    """수천 자리 환각 숫자에서 파서가 죽지 않아야 한다 (제출 경로 보호).

    Python 3.11+의 int(str) 4,300자리 제한 때문에 과거 ValueError로 중단됐다.
    실제 문샷(최대 7,874자리)·mix(15,634자리) 생성물에 존재하는 패턴.
    """
    huge = "The answer is \\boxed{" + "9" * 8000 + "}"
    result = extract_answer(huge)          # 죽지 않는 것이 핵심
    assert result is None or isinstance(result, int)


def test_normal_answers_unaffected_by_digit_limit_change():
    """상한 해제가 기존 판정을 바꾸지 않아야 한다."""
    assert extract_answer("So the answer is \\boxed{42}.") == 42
    assert extract_answer("final answer is -7") == -7


# --- 지수 폭탄 회귀 테스트 (2026-08-16 리허설에서 실제로 D-Day를 막을 뻔한 버그) ---
#
# `1.00e23610081082016` 같은 문자열이 `Fraction(10) ** 23_610_081_082_016` 을 유발해
# 831문항 파이프라인이 36.7% 지점에서 무한 정지했다. 상한이 없으면 제출 자체가 불가능하다.

def test_exponent_bomb_returns_none_fast():
    """거대 지수는 즉시 None. 시간 안에 끝나야 한다 (정지 회귀 방지)."""
    import time
    for s in ("1.00e23610081082016", "3.00e7873360360640", "1e999999999", "-2.5e10001"):
        t = time.time()
        assert normalize_to_int(s) is None, s
        assert time.time() - t < 0.5, f"{s} 처리에 {time.time()-t:.2f}초 — 상한이 안 걸렸다"


def test_exponent_within_bound_still_works():
    """상한 이내는 기존 동작 그대로 — 기존 점수가 바뀌지 않음을 보장한다."""
    assert normalize_to_int("1e3") == 1000
    assert normalize_to_int("3.4e15") == 3_400_000_000_000_000
    assert normalize_to_int("-1.5e2") == -150
    assert normalize_to_int("2.5e1") == 25
    assert normalize_to_int("1e-3") is None        # 정수 아님
    assert normalize_to_int("1e10000") == 10 ** 10000    # 경계 안쪽은 계산한다
    assert normalize_to_int("1e10001") is None           # 경계 밖은 버린다


def test_extract_answer_survives_exponent_bomb():
    """생성물 안에 폭탄이 섞여 있어도 파이프라인이 진행돼야 한다."""
    import time
    text = ("Some reasoning with 1.00e23610081082016 and 3.00e7873360360640 "
            "appearing repeatedly. " * 20) + r"The final answer is \boxed{42}."
    t = time.time()
    assert extract_answer(text) == 42
    assert time.time() - t < 1.0


"""학습 전용 엄격 검증기 테스트.

제출 파서(test_parse.py)와 목적이 정반대임에 주의:
  제출 파서 = 불완전한 출력에서도 답을 **회수**해야 하므로 관대
  엄격 검증기 = 학습 데이터에 오염을 **들이지 않아야** 하므로 보수적
따라서 같은 입력에 대해 두 모듈이 다른 결과를 내는 것이 정상이다.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.strict_verify import classify, is_trainable, strict_answer  # noqa: E402


def box(inner: str) -> str:
    return f"Therefore the answer is \\boxed{{{inner}}}."


# --- 통과해야 하는 것 ---

@pytest.mark.parametrize("inner,expected", [
    ("42", 42),
    ("-7", -7),
    ("0", 0),
    (" 42 ", 42),              # 양끝 여백은 허용
    ("1,234", 1234),           # 천단위 구분
    ("-1,234,567", -1234567),
    ("1\\,234", 1234),         # LaTeX 얇은공백 천단위
    ("\\!42\\!", 42),          # 순수 간격 매크로
    ("−7", -7),                # 유니코드 마이너스
])
def test_strict_accepts_plain_integers(inner, expected):
    assert strict_answer(box(inner)) == expected


def test_strict_accepts_latex_display_close():
    """`\\[ \\boxed{2}\n\\]` 형태는 정상 종결이다 (과거 38.2% 오탈락 원인)."""
    assert strict_answer("\\[ \\boxed{2}\n\\]") == 2


# --- 반드시 거부해야 하는 것 ---

@pytest.mark.parametrize("inner", [
    "1 2",        # 내부 공백 = 답 두 개 나열 가능성 (12로 둔갑 금지)
    "1,2,3",      # 천단위가 아닌 콤마 나열 (123으로 둔갑 금지)
    "1{2}",       # 중괄호 잔재 (12로 둔갑 금지)
    "\\sqrt{10028}",
    "18\\sqrt3",
    "\\frac{25}{9}",
    "integer",
    "d",
    "40\\%",
    "\\infty",
    "2^k",
    "",
])
def test_strict_rejects_non_integer_boxes(inner):
    assert strict_answer(box(inner)) is None


def test_strict_requires_boxed():
    """fallback 없음 — 결론 문장만으로는 통과하지 못한다."""
    assert strict_answer("The final answer is 42.") is None


def test_strict_uses_last_box():
    assert strict_answer("\\boxed{1} ... 다시 계산하면 \\boxed{2}") == 2


# --- 형식 적격성 (답의 정오와 무관) ---

def test_hangul_solution_not_trainable():
    ok, why = is_trainable("계산하면 답은 \\boxed{7}")
    assert not ok and why == "hangul"


def test_huge_number_not_trainable():
    ok, why = is_trainable("\\boxed{" + "9" * 60 + "}")
    assert not ok and why == "huge_number"


def test_repetition_not_trainable():
    chunk = "Now we compute the value of the expression carefully step by step. "
    ok, why = is_trainable(chunk * 5 + "\\boxed{3}")
    assert not ok and why == "repetition"


# --- 3분류 (격리는 폐기가 아니다) ---

def test_classify_strict():
    assert classify(box("42"), 42) == "strict"


def test_classify_quarantine_when_only_loose_parser_agrees():
    """엄격 기준 미달이나 관대한 파서로는 정답 → 폐기가 아니라 격리."""
    assert classify(box("\\sqrt{10028}"), 10028) == "quarantine"
    assert classify("The final answer is 42.", 42) == "quarantine"


def test_classify_reject_on_wrong_answer():
    assert classify(box("41"), 42) == "reject"


def test_classify_reject_on_format_defect_even_if_correct():
    """형식 결함은 답이 맞아도 학습 부적격."""
    assert classify("답은 \\boxed{42}", 42) == "reject"

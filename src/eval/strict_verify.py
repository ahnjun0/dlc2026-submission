"""학습 데이터 전용 **엄격** 정답 검증기 — 제출용 파서와 의도적으로 분리한다.

왜 분리하는가:
  제출 파서(`src/eval/parse.py`)는 불완전한 출력에서도 답을 회수해야 하므로
  관대해야 한다 (boxed 실패 시 결론 패턴 → 마지막 숫자 fallback).
  그러나 **학습 데이터 검증기가 관대하면 오염을 그대로 학습**한다.

실측 근거 (2026-08-06):
  sft_v2 19,185건 중 746건(3.9%)이 관대한 파서 덕에 '정답'으로 채택됐다.
    - boxed 안이 정수가 아님 335건: `\\boxed{\\sqrt{10028}}`→10028,
      `\\boxed{integer}`, `\\boxed{d}`, `\\boxed{40\\%}`, `\\boxed{\\frac{25}{9}}`
    - boxed 자체가 없어 마지막 숫자 fallback으로 통과 411건
  이런 풀이는 "최종 답을 boxed 정수로 쓴다"는 형식 자체를 망가뜨린다.

정책: **버리지 않고 격리(quarantine)한다.** 3.9% 중 실제 정답도 섞여 있으므로
      폐기하면 손해다. 별도 파일로 분리해 두고 필요 시 심사한다.
"""

from __future__ import annotations

import re

_BOX = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_INT = re.compile(r"^-?\d+$")
# 천단위 구분 형태만 허용: 1,234 / -1,234,567 (1,2,3 같은 '나열'은 불허)
_THOUSANDS = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")
# 값에 영향이 없는 순수 간격 매크로만 제거 (콤마·공백·중괄호는 제거하지 않는다)
_SPACING = ("\\!", "\\;", "\\quad", "\\qquad", "$", "\\$")

# 학습 부적격 신호: 반복 붕괴·미종료·거대정수 환각
_REPEAT = re.compile(r"(.{40,}?)\1{3,}", re.S)   # 같은 40자+ 덩어리가 4회 이상
_HUGE = re.compile(r"\d{50,}")                    # 50자리 이상 = 환각 (gold 최대 16자리)
_HANGUL = re.compile(r"[가-힣]")


def strict_answer(solution: str) -> int | None:
    """마지막 \\boxed{...} 안이 **순수 정수 하나**일 때만 그 값을 반환한다.

    fallback 없음이 핵심. boxed가 없거나 내용이 단일 정수가 아니면 None.

    정규화에서 의도적으로 **하지 않는** 것 (하면 서로 다른 값이 같아져 버린다):
      - 내부 공백 제거 → `1 2`가 12로 둔갑 (답 두 개를 나열한 것일 수 있음)
      - 콤마 무조건 제거 → `1,2,3`이 123으로 둔갑 (천단위 형태만 허용한다)
      - 중괄호 제거 → `1{2}`가 12로 둔갑 (수식 잔재이지 정수가 아님)
    """
    boxes = _BOX.findall(solution)
    if not boxes:
        return None
    inner = boxes[-1]
    for d in _SPACING:
        inner = inner.replace(d, "")
    inner = inner.replace("\\,", ",")  # LaTeX 얇은공백은 천단위 구분자로 쓰인다
    inner = inner.replace("−", "-").strip()  # 유니코드 마이너스 + 양끝 여백만
    if "{" in inner or "}" in inner:
        return None                    # 수식 잔재 (\sqrt{...}, 1{2} 등)
    if re.search(r"\s", inner):
        return None                    # 내부 공백 = 단일 정수가 아님
    if _THOUSANDS.match(inner):
        inner = inner.replace(",", "")
    return int(inner) if _INT.match(inner) else None


def is_trainable(solution: str) -> tuple[bool, str]:
    """학습에 넣어도 되는 풀이인지. (적격여부, 사유) 반환.

    답의 정오와 무관한 **형식 결함**만 본다. 정오 판정은 strict_answer로.
    """
    if _HANGUL.search(solution):
        return False, "hangul"          # v4 교훈: 한국어 사고 혼입 기각
    if _REPEAT.search(solution):
        return False, "repetition"      # 반복 붕괴
    if _HUGE.search(solution):
        return False, "huge_number"     # 거대정수 환각
    # 잘림(미종료)은 별도 휴리스틱을 두지 않는다: 문장 끝 문자로 판정하려던 시도는
    # `\boxed{2}\n\\]` 처럼 정상 LaTeX 종결을 잘림으로 오판했다(19,185건 중 38.2%).
    # 실제로 잘린 생성은 boxed가 없어 strict_answer에서 자연히 걸러진다.
    return True, "ok"


def classify(solution: str, gold: int) -> str:
    """채택 판정: 'strict' | 'quarantine' | 'reject'.

    strict     — 엄격 검증 통과 + 형식 적격 → 학습 사용
    quarantine — 관대한 파서로는 정답이나 엄격 기준 미달 → 보류 (폐기 아님)
    reject     — 답이 틀렸거나 형식 결함
    """
    ok, _ = is_trainable(solution)
    if not ok:
        return "reject"
    if strict_answer(solution) == gold:
        return "strict"
    # 관대한 파서로는 맞는가? (그렇다면 폐기가 아니라 격리)
    from src.eval.parse import extract_answer

    if extract_answer(solution) == gold:
        return "quarantine"
    return "reject"

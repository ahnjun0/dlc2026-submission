"""모델 출력에서 최종 정수 답을 추출한다.

우선순위:
  1. 마지막 \\boxed{...} 내용
  2. "final answer is ...", "answer: ...", "답: ..." 등 결론 패턴의 마지막 등장
  3. 텍스트 전체의 마지막 숫자 (fallback)

각 후보는 normalize_to_int()로 정수화를 시도하고, 실패하면 다음 후보로 넘어간다.
반환값 None은 "추출 실패"이며 채점에서 오답 처리된다.

주의: 답 분포에 음수 3%, 0, 1e6 이상 거대 정수가 실존하므로 (train EDA 확인)
부호를 버리거나 float 경유로 정밀도를 잃으면 안 된다.
"""

from __future__ import annotations

import re
import sys
from fractions import Fraction

# Python 3.11+는 int(str) 변환을 4,300자리로 제한한다. 모델이 환각으로 수천 자리
# 숫자를 뱉는 경우가 실존하므로(문샷 최대 7,874자리 / mix 15,634자리) 상한을 풀지
# 않으면 파서가 ValueError로 죽는다 = 제출 경로 전체가 중단된다.
# 상한만 올리고 판정 로직은 그대로 둔다: 거대 환각값은 어차피 gold(최대 16자리)와
# 불일치하므로 '오답 한 표'로 처리되어 다수결에서 자연히 탈락한다.
sys.set_int_max_str_digits(200000)

# 결론 패턴 (영/한). 뒤에 오는 짧은 구절을 캡처해 그 안에서 숫자를 찾는다.
_CONCLUSION_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|answer\s*[:=]|answer\s*was|정답은|정답\s*[:=]|답은|답\s*[:=])"
    r"[^\n]{0,80}",
    re.IGNORECASE,
)

_BOXED_START_RE = re.compile(r"\\boxed\s*\{")

# 정수/소수/분수/과학표기 (콤마 천단위, 유니코드 마이너스 허용)
_NUMBER_RE = re.compile(
    r"[-−]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
_FRAC_RE = re.compile(r"([-−]?)\s*\\[dt]?frac\s*\{\s*([-−]?\d+)\s*\}\s*\{\s*([-−]?\d+)\s*\}")


def _extract_boxed(text: str) -> list[str]:
    """모든 \\boxed{...} 내용을 중괄호 균형을 맞춰 추출한다 (중첩 허용)."""
    out = []
    for m in _BOXED_START_RE.finditer(text):
        depth, start = 1, m.end()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:i])
                    break
    return out


def normalize_to_int(s: str) -> int | None:
    """숫자 문자열 하나를 정수로 정규화. 정수가 아니면 None."""
    s = s.strip().replace("−", "-").replace(" ", "")
    s = s.replace(",", "")  # 천단위 콤마
    if not s:
        return None
    # 순수 정수는 float를 거치지 않는다 (거대 정수 정밀도 보존)
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        f = Fraction(s)
        return int(f) if f.denominator == 1 else None
    if re.fullmatch(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", s):
        exp = int(re.split(r"[eE]", s)[1])
        # **지수 상한 — 무한 정지 방지** (2026-08-16 리허설에서 실제로 걸렸다)
        #
        # `_NUMBER_RE`의 지수부가 `\d+`로 무제한이라, 모델이 뱉은
        # `1.00e23610081082016` 같은 문자열에서 `Fraction(10) ** 23_610_081_082_016`
        # 을 계산하려 들어 **영원히 끝나지 않는다.** 리허설 3단계가 831문항 중
        # 305번째(36.7% 지점)에서 90분간 CPU 100%로 정지했고, 그대로였다면
        # **D-Day에 제출 자체가 불가능**했다. 같은 패턴이 그 파일에 165건 있었다.
        #
        # 상한 10,000은 극도로 보수적이다 — 우리 gold의 최대 절댓값은 3.4e15이고,
        # 10^10000을 넘는 값은 어떤 문항의 답도 될 수 없다. 따라서 이 가드는
        # **기존 점수를 바꿀 수 없고**, 지금까지 답을 내던 입력의 결과도 동일하다.
        # (버리는 입력들은 답을 내던 게 아니라 계산이 끝나지 않던 것들이다.)
        if abs(exp) > 10_000:
            return None
        f = Fraction(s.lower().replace("e", "E").split("E")[0]) * Fraction(10) ** exp
        return int(f) if f.denominator == 1 else None
    return None


def _candidate_to_int(chunk: str) -> int | None:
    """텍스트 조각(boxed 내용 또는 결론 구절)에서 정수 하나를 뽑는다."""
    # LaTeX 분수 우선 처리
    fm = _FRAC_RE.search(chunk)
    if fm:
        sign = -1 if fm.group(1) in ("-", "−") else 1
        num = int(fm.group(2).replace("−", "-"))
        den = int(fm.group(3).replace("−", "-"))
        if den != 0 and num % den == 0:
            return sign * (num // den)
        return None
    chunk = re.sub(r"\\[a-zA-Z]+", " ", chunk)  # 잔여 LaTeX 명령 제거
    chunk = chunk.replace("$", " ")
    for m in _NUMBER_RE.finditer(chunk):  # 조각 안에서는 첫 숫자가 결론일 확률이 높음
        cand = m.group(0)
        v = normalize_to_int(cand)
        if v is not None:
            return v
    return None


def extract_answer(text: str) -> int | None:
    """모델 출력 전체에서 최종 정수 답을 추출한다."""
    if not text:
        return None

    # 1) 마지막 boxed부터 역순으로
    for chunk in reversed(_extract_boxed(text)):
        v = _candidate_to_int(chunk)
        if v is not None:
            return v

    # 2) 결론 패턴 — 마지막 등장부터 역순으로
    for m in reversed(list(_CONCLUSION_RE.finditer(text))):
        v = _candidate_to_int(m.group(0))
        if v is not None:
            return v

    # 3) fallback: 텍스트의 마지막 숫자
    nums = [m.group(0) for m in _NUMBER_RE.finditer(text)]
    for cand in reversed(nums):
        v = normalize_to_int(cand)
        if v is not None:
            return v
    return None

"""`scripts/lib/vllm_env.sh` 와 D-Day 스크립트의 인라인 사본이 어긋나지 않는지 지킨다.

D-Day 스크립트는 2,000문항으로 검증된 뒤라 리팩터링하지 않기로 했고(인라인 유지),
대신 **두 곳이 같은 환경을 내보내는지**를 테스트로 고정한다.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING",
        "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "VLLM_USE_FLASHINFER_SAMPLER"]


def _exported(text: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"export\s+([A-Z0-9_]+)=", text)}


def test_dday_script_exports_same_env_keys():
    lib = (ROOT / "scripts/lib/vllm_env.sh").read_text()
    dday = (ROOT / "scripts/run_dday_cascade.sh").read_text()
    for k in KEYS:
        assert k in _exported(lib), f"{k} 가 공유 환경 파일에 없다"
        assert k in _exported(dday), f"{k} 가 D-Day 스크립트에 없다 — 두 곳이 어긋났다"


def test_blackwell_detection_present_in_both():
    for p in ("scripts/lib/vllm_env.sh", "scripts/run_dday_cascade.sh"):
        t = (ROOT / p).read_text()
        assert "compute_cap" in t and "VLLM_USE_FLASHINFER_SAMPLER=0" in t, \
            f"{p} 에 Blackwell 자동 감지가 없다"

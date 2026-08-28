# vLLM 실행 환경 — **모든 생성 스크립트가 이 파일을 source 한다.**
#
# 왜 파일로 뺐는가 (2026-08-27): 이 블록이 스크립트마다 복붙돼 있었고, 그래서
#   · 8/26 리허설: `run_dday_cascade.sh` 에만 Blackwell 대응이 빠져 1단계가 즉사
#   · 8/27 exp-122: 새로 쓴 스크립트에 다시 빠져 생성이 즉사 (같은 실수의 반대 방향)
# 두 번 같은 사고가 났으면 원인은 «잊었다»가 아니라 **«잊을 수 있는 구조»** 다.
#
# 사용:  source scripts/lib/vllm_env.sh
#
# ⚠ `scripts/run_dday_cascade.sh` 는 **인라인 사본을 유지한다** — 2,000문항으로
#    end-to-end 검증된 스크립트를 D-Day 직전에 리팩터링하지 않는다.
#    두 곳이 어긋나지 않는지는 `tests/test_vllm_env.py` 가 지킨다.

export VLLM_BATCH_INVARIANT=1          # 실행 간 잡음 0 — 작은 효과 판정의 선행 조건
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# **Blackwell(SM 12.x) 자동 감지** — 하드코딩하지 않는다.
# Ampere 에서는 FlashInfer 가 정상이고 더 빠르다.
if command -v nvidia-smi >/dev/null 2>&1; then
  _CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  case "${_CC%%.*}" in
    12|11) export VLLM_USE_FLASHINFER_SAMPLER=0
           echo "[env] compute_cap ${_CC} (Blackwell 계열) → VLLM_USE_FLASHINFER_SAMPLER=0" ;;
    *)     echo "[env] compute_cap ${_CC:-미상} → FlashInfer 샘플러 기본값 유지" ;;
  esac
fi

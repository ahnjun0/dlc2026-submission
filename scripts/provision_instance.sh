#!/bin/bash
# **신규 GPU 인스턴스 프로비저닝** — 한 번에 가동 상태까지.
#
# 왜: 절차는 `docs/dday-protocol.md` 1절에 상세히 적혀 있으나 **실행 스크립트가 없었다**.
# 2026-08-25 03:40 UTC 에 두 인스턴스가 동시에 다운되면서 이 구멍이 드러났다.
# D-Day 당일 또는 인스턴스 소멸 시 **6단계를 손으로 치는 상황**을 없앤다.
#
# 사용:  rsync 로 코드를 올린 뒤   bash scripts/provision_instance.sh
#        (HF 토큰이 필요하면 HF_TOKEN 환경변수로 넘긴다)
set -u
# 기본값은 종전 값 그대로. 심사자는 DLC_ROOT 로 덮어쓴다.
DLC_ROOT=${DLC_ROOT:-/workspace/dlc}
cd "$DLC_ROOT" 2>/dev/null || { echo "치명: $DLC_ROOT 가 없다 (DLC_ROOT 로 지정할 것)"; exit 1; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# **인터프리터를 탐지한다** — `python` 이 PATH 에 없는 호스트가 흔하다.
# 2026-08-25 실측: 이 스크립트 자신이 ⑤ 검증에서 그 이유로 죽었다
# (run_dday_cascade.sh 에는 같은 수정을 이미 했는데 여기엔 안 했다).
PY=""
for c in /venv/main/bin/python /venv/dlc/bin/python "$(command -v python3)" "$(command -v python)"; do
  [ -x "$c" ] && { PY=$c; break; }
done
[ -x "$PY" ] || { echo "치명: 파이썬을 찾지 못했다"; exit 1; }
log "PY=$PY"
FAIL=0

log "① 회선 실측 (실존 파일로 — 404 를 재면 가짜 저속이 나온다)"
S=$(curl -sLo /dev/null -w "%{speed_download}" --max-time 15 \
  https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz 2>/dev/null || echo 0)
log "   PyPI $(awk -v s="$S" 'BEGIN{printf "%.1f", s/1048576}') MB/s"
awk -v s="$S" 'BEGIN{exit !(s < 5242880)}' && log "   ⚠ 5MB/s 미만 — 다른 호스트를 고려할 것"

# **딸려온 로컬 .venv 를 제거한다** — rsync 로 맥의 .venv 가 넘어오면 uv 가 그것을
# 대상으로 삼고 깨진 심볼릭 링크에서 죽는다 (2026-08-25 실측).
[ -e .venv ] && { rm -rf .venv; log "   딸려온 .venv 제거"; }
log "② 패키지 (문서화된 순서·함정 반영)"
uv pip install -q --python "$PY" vllm==0.26.0 trl==1.9.2 peft==0.20.0 datasets==5.0.1 \
  accelerate==1.14.0 bitsandbytes==0.50.0 math-verify==0.9.0 pytest || FAIL=1
# vllm 은 CUDA13 빌드다. --reinstall-package 없으면 uv 가 +cu128 을 "만족"으로 보고 건너뛴다
# **셋 다 --reinstall-package 를 걸어야 한다** — torch 만 걸면 torchvision 이 cu128 로 남아
# "PyTorch has CUDA 13.0 and torchvision has CUDA 12.8" 로 죽는다 (2026-08-25 실측).
uv pip install -q --python "$PY" \
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130 || FAIL=1
uv pip uninstall -q --python "$PY" torchcodec 2>/dev/null   # 버전 불일치로 import 실패 — 우리는 안 씀
uv pip install -q --python "$PY" transformers==5.14.1        # 5.15 는 warmup_ratio 제거 → sft_lora.py TypeError
# **`kernels` 는 FlashAttention 로더의 필수 의존**이다 (2026-08-26 실측).
# 없으면 `--attn kernels-community/flash-attn2` 학습이 ImportError 로 죽는다.
# 추론에는 없어도 되지만 **학습·재현 검증(규칙 8.2)에는 반드시 필요**하다.
uv pip install -q --python "$PY" "kernels>=0.15.2,<0.16.0"

log "③ 결정성 환경변수 (규칙 8.2 — 재현 불가 시 수상 취소)"
cat > /workspace/dlc/.dlc_env <<'ENVEOF'
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ENVEOF
grep -q dlc_env ~/.bashrc || echo "source /workspace/dlc/.dlc_env" >> ~/.bashrc
# Blackwell(sm120)은 flashinfer 샘플러가 능력 탐지에 실패한다 (8/24 실측)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "blackwell|PRO 6000|B200|5090"; then
  echo "export VLLM_USE_FLASHINFER_SAMPLER=0" >> /workspace/dlc/.dlc_env
  log "   Blackwell 감지 → VLLM_USE_FLASHINFER_SAMPLER=0 추가"
fi

log "④ D-Day 가중치 회수 (HF)"
"$PY" - <<'PYX' || FAIL=1
import os, shutil
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
want = {"exp-004-v2-adapter": "/workspace/ckpt/exp-004_star_v2",      # 주 모델
        "exp-035-contest-lora": "/workspace/ckpt/exp-035_contest",     # 3단계 증원
        "exp-015-moonshot/ep1": "/workspace/ckpt/moonshot_ep1"}        # 파트너 (12GB)
for src, dst in want.items():
    if os.path.exists(os.path.join(dst, "adapter_model.safetensors")) or \
       os.path.exists(os.path.join(dst, "model.safetensors")):
        print(f"  이미 존재: {dst}"); continue
    p = snapshot_download(os.environ.get("DLC_WEIGHTS_REPO","ahnjun0/dlc2026-weights"), allow_patterns=f"{src}/*",
                          local_dir="/workspace/_dl", token=tok, max_workers=4)
    os.makedirs(dst, exist_ok=True)
    s = os.path.join("/workspace/_dl", src)
    for f in os.listdir(s):
        shutil.move(os.path.join(s, f), dst)   # 복사 아닌 이동 — 디스크 이중 점유 방지
    print(f"  회수: {dst}")
shutil.rmtree("/workspace/_dl", ignore_errors=True)
PYX

log "⑤ 검증"
"$PY" -c "import torch,torchvision,vllm; from vllm import LLM; print('   import OK · torch',torch.__version__)" || FAIL=1
"$PY" -c "
import sys; sys.path.insert(0,'.')
from src.eval.parse import extract_answer
assert extract_answer(r'answer is \boxed{42}')==42
assert extract_answer('1.00e23610081082016') is None, '지수 상한 가드가 없다'
print('   파서 가드 OK')" || FAIL=1
"$PY" -m pytest tests/ -q 2>&1 | tail -2 || FAIL=1
df -h /workspace | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ "$FAIL" -eq 0 ]; then log "**PROVISION_OK** — 가동 준비 완료"; else log "**PROVISION_FAILED** — 위 오류 확인"; exit 1; fi

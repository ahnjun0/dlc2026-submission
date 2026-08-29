#!/bin/bash
# exp-035 2~4단계 무인 연쇄: frontier 재확인 → 빌드 → 학습 → 백업.
#
# 레시피는 **v2와 동일하게 고정**한다 (clean v2 실측 커맨드 그대로):
#   LoRA r128 all-linear / LR 2e-5 / 2 epoch / seq 4096 / flash-attn2
# 유일한 변수는 **문제 풀 구성**이다. 다른 것을 건드리면 판정이 흐려진다.
set -x
source /venv/main/bin/activate
# **경로를 환경변수로** (2026-08-28) — 기본값은 종전 값 그대로다.
DLC_ROOT=${DLC_ROOT:-/workspace/dlc}
DLC_CKPT=${DLC_CKPT:-/workspace/ckpt}
export HF_HOME=${HF_HOME:-/workspace/hf}
cd "$DLC_ROOT"
OUT=experiments/exp-035_contest_star
CKPT=$DLC_CKPT/exp-035_contest

# --- 2단계: frontier 독립 시드 재확인 (4,480문항 × 8샘플, 시드 43) ---
T=$(date +%s)
python src/inference/generate.py \
  --input data/processed/contest_frontier.csv \
  --output ${OUT}/frontier_samp8_s43.jsonl \
  --lora $DLC_CKPT/exp-004_star_v2 \
  --n 8 --seed 43 --chunk 500 --resume || exit 1
echo "RECHECK_DONE $(( ($(date +%s)-T)/60 ))분  rows=$(wc -l < ${OUT}/frontier_samp8_s43.jsonl)"

# --- 3단계: 데이터 빌드 (경시 전용 — v2 데이터와 혼합하지 않는다) ---
python src/data/build_sft_contest.py \
  --gens ${OUT}/contest_samp8_s42.jsonl \
  --recheck-gens ${OUT}/frontier_samp8_s43.jsonl \
  --output data/processed/sft_contest.jsonl || exit 1
wc -l data/processed/sft_contest.jsonl

# --- 4단계: 학습 (v2 레시피 고정) ---
T=$(date +%s)
python src/train/sft_lora.py \
  --data data/processed/sft_contest.jsonl \
  --output ${CKPT} \
  --lr 2e-5 --epochs 2 --seed 42 \
  --attn kernels-community/flash-attn2
echo "TRAIN_SEC=$(( $(date +%s)-T ))"

# **종료코드가 아니라 산출물로 판정한다** (8/8·8/9 두 번 데임).
# 저장 경로가 스크립트마다 다르므로 고정 경로 대신 find로 찾는다 (8/9 가드 경로 버그 교훈).
ADAPTER=$(find ${CKPT} -name adapter_model.safetensors | head -1)
if [ -z "${ADAPTER}" ]; then echo "TRAIN_FAILED_NO_ARTIFACT"; ls -laR ${CKPT}; exit 1; fi
echo "ARTIFACT_OK ${ADAPTER}"

# --- bf16 변환 후 HF 백업 (보관은 전부 bf16) ---
python - <<PY
import torch, os
from safetensors.torch import load_file, save_file
from huggingface_hub import HfApi
src = os.path.dirname("${ADAPTER}")
sd = load_file("${ADAPTER}")
sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
save_file(sd, "${ADAPTER}", metadata={"format": "pt"})
print("bf16:", os.path.getsize("${ADAPTER}")/1e6, "MB")
api = HfApi()
api.upload_folder(folder_path=src, repo_id="ahnjun0/dlc-artifacts", repo_type="model",
                  path_in_repo="exp-035-contest-lora",
                  ignore_patterns=["README.md", "optimizer.pt", "checkpoint-*"])
got = [f for f in api.list_repo_files("ahnjun0/dlc-artifacts") if f.startswith("exp-035-contest-lora/")]
print("BACKUP_OK" if any("adapter_model.safetensors" in f for f in got) else "BACKUP_INCOMPLETE", got)
PY
df -h "${DLC_TMP:-/workspace}" | tail -1
echo CONTEST_PIPELINE_DONE

#!/bin/bash
# **D-Day 캐스케이드 파이프라인** (exp-101 채택 구성 · 2026-08-24)
#
# LB 짝 검정 실측: 668/831 vs 전량 풀링 대조군 665 = **+3문항**, 동시에 시간 **반감**.
#
# 종전 `run_dday_rehearsal.sh` 와의 차이 — **단계 간 의존이 생겼다**:
#   1단계 v2 전량        → 확정(최다>=30)분은 여기서 끝. 나머지만 2단계로
#   2단계 문샷 (미확정분) → 풀링 격차<=5 인 접전분만 3단계로
#   3단계 경시STaR (접전분) → 최종 조립
# LB 실측 분기: 1단계 확정 462(55.6%) · 3단계 증원 107(12.9%)
#
# **각 단계 끝에 제출물을 만든다** (운영진 8/21 답변: 재제출 가능).
#   sub_1_v2only  ~1시간   ← 이 시점에 점수가 확보된다. 이후는 전부 개선 시도다
#   sub_2_cascade_no3         3단계만 뺀 캐스케이드 (**전량 풀링이 아님** — 확정분은 v2 단독)
#   sub_3_cascade             채택 구성 = 최종 제출본
#
# ⚠ **결정성**: 청크 100 고정 + 결정성 환경변수. 단 2·3단계는 입력이 부분집합이라
#    청크 경계가 전량 실행과 다르다 — **캐스케이드끼리는 재현되지만 전량 풀링과는
#    문항별로 다를 수 있다**. 이는 파이프라인 정의상 당연하며 결함이 아니다.
set -o pipefail
export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# **Blackwell(SM 12.x) 자동 감지** — 2026-08-26 리허설이 잡은 D-Day 블로커.
# 다른 스크립트엔 전부 넣어둔 `VLLM_USE_FLASHINFER_SAMPLER=0` 이 **정작 본 스크립트에만
# 빠져 있었다**. RTX PRO 5000 에서 1단계 엔진 초기화가 즉시 실패했다
# (`Failed to get device capability: SM 12.x requires CUDA >= 12.9` → FlashInfer 샘플러 선택 → RuntimeError).
# 장비를 못 고르는 D-Day 에 이게 남아 있으면 **00:00 에 파이프라인이 안 뜬다.**
# 하드코딩하지 않고 감지한다 — Ampere 에서는 FlashInfer 가 정상이고 더 빠르다.
if command -v nvidia-smi >/dev/null 2>&1; then
  _CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  case "${_CC%%.*}" in
    12|11) export VLLM_USE_FLASHINFER_SAMPLER=0
           echo "[env] compute_cap ${_CC} (Blackwell 계열) → VLLM_USE_FLASHINFER_SAMPLER=0" ;;
    *)     echo "[env] compute_cap ${_CC:-미상} → FlashInfer 샘플러 기본값 유지" ;;
  esac
fi

cd /workspace/dlc
IN=${DDAY_IN:-data/raw/deep_chal_math_leaderboard_filtered.csv}   # D-Day 엔 test.csv
OUT=${DDAY_OUT:-experiments/dday_cascade}

# **단계마다 자동 백업한다** (2026-08-26 신설 — 인스턴스 만료로 GPU 7시간치를 잃고 나서).
# 정책은 CLAUDE.md 에 있었고 «사람이 기억해서 하는 백업»으로 뒀더니 실패했다.
# D-Day 는 12시간을 생성하므로, 도중에 컨테이너가 죽으면 **정확히 같은 방식으로** 전부 날아간다.
# 실패해도 파이프라인을 멈추지 않는다 — 백업은 보험이지 관문이 아니다.
DDAY_BACKUP=${DDAY_BACKUP:-1}
backup_stage () {   # $1 = HF 하위 경로 꼬리표
  [ "${DDAY_BACKUP}" = "1" ] || return 0
  [ -s ~/.cache/huggingface/token ] || { echo "[backup] HF 토큰 없음 — 건너뜀"; return 0; }
  echo "[backup] $1 업로드 시작 (비차단)"
  ( $PY - "$1" <<'PYB'
import os, sys, glob
from huggingface_hub import HfApi
tag = sys.argv[1]
out = os.environ.get("DDAY_OUT", "experiments/dday_cascade")
api = HfApi(token=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
files = [f for f in glob.glob(out + "/*") if os.path.isfile(f)]
if not files:
    print("[backup] 올릴 파일 없음"); sys.exit(0)
api.upload_folder(folder_path=out, path_in_repo=f"dday/{tag}", repo_id="ahnjun0/dlc-artifacts")
remote = [f for f in api.list_repo_files("ahnjun0/dlc-artifacts") if f.startswith(f"dday/{tag}/")]
print(f"[backup] {tag}: 로컬 {len(files)} → 원격 {len(remote)} 확인")
PYB
  ) || echo "[backup] $1 실패 — **비치명, 계속 진행**"
}
export DDAY_OUT
# **기본값 22** (2026-08-28 배선). 종전 30 에서 낮췄고 **출력은 안 바뀐다** —
# LB 831 · val 460×3시드 · **리허설 2,000문항** 합계 **4,211 사례에서 답 변경 0**.
# 선별기(cascade_select)와 조립기(build_cascade)가 같은 집합을 보는 것도 확인했다(차집합 0).
# 얻는 것: 2단계 대상 1,189 → 844문항, 총 16.20h → **12.15h**.
# 되돌리려면 `DDAY_CONFIRM=30`. 근거: docs/confirm-threshold-2026-08-27.md
CONFIRM=${DDAY_CONFIRM:-22}
MARGIN=${DDAY_MARGIN:-5}
# **파이썬 경로 자동 탐지** — 호스트마다 venv 이름이 다르다.
# 2026-08-25 실측: 4090 은 `/venv/main`, PRO 6000 은 `/venv/dlc` 라 `python` 이 PATH 에 없어
# **D-Day 스크립트가 1단계에서 즉사**했다 (2,000문항 리허설이 잡아냈다).
PY=${DDAY_PY:-}
if [ -z "$PY" ]; then
  for c in /venv/main/bin/python /venv/dlc/bin/python "$(command -v python3)" "$(command -v python)"; do
    [ -x "$c" ] && { PY=$c; break; }
  done
fi
[ -x "$PY" ] || { echo "치명: 파이썬을 찾지 못했다"; exit 1; }
echo "PY=$PY"
mkdir -p ${OUT}
T0=$(date +%s)
echo "입력 ${IN} · 출력 ${OUT} · 문턱 confirm=${CONFIRM} margin=${MARGIN}"

# ---------- 1단계: v2 전량 ----------
echo "=== [1/6] v2 전량 생성 ($(date +%H:%M:%S)) ==="
$PY src/inference/generate.py --input ${IN} \
  --output ${OUT}/v2_samp32.jsonl \
  --lora /workspace/ckpt/exp-004_star_v2 --n 32 --seed 42 \
  --chunk 100 --resume || { echo "STAGE1_FAILED"; exit 1; }
T1=$(date +%s); echo "STEP1_SEC=$((T1-T0))"

echo "=== [2/6] **보험 제출물** v2 단독 ($(date +%H:%M:%S)) ==="
$PY src/inference/make_submission.py \
  --gens ${OUT}/v2_samp32.jsonl --restrict-to-expect \
  --output ${OUT}/sub_1_v2only.csv --expect-ids ${IN} || { echo "SUB1_FAILED"; exit 1; }
echo "SUB1_READY — 이 시점부터 제출물이 존재한다"
backup_stage stage1

# ---------- 2단계: 문샷 (미확정분만) ----------
echo "=== [3/6] 2단계 대상 선별 ($(date +%H:%M:%S)) ==="
$PY src/eval/cascade_select.py --stage 2 --input ${IN} \
  --v2 ${OUT}/v2_samp32.jsonl --confirm ${CONFIRM} \
  --out ${OUT}/stage2_input.csv || { echo "SELECT2_FAILED"; exit 1; }

if [ ! -s /workspace/ckpt/moonshot_ep1/model.safetensors ]; then
  echo "문샷 가중치 회수 중 (12GB)"
  $PY - <<'PYX'
from huggingface_hub import snapshot_download
import shutil, os
snapshot_download("ahnjun0/dlc-artifacts", allow_patterns=["exp-015-moonshot/ep1/*"],
                  local_dir="/workspace/dl_tmp", max_workers=4)
os.makedirs("/workspace/ckpt/moonshot_ep1", exist_ok=True)
src = "/workspace/dl_tmp/exp-015-moonshot/ep1"
for f in os.listdir(src):
    shutil.move(os.path.join(src, f), "/workspace/ckpt/moonshot_ep1/")   # 이동 (디스크 이중 점유 방지)
shutil.rmtree("/workspace/dl_tmp", ignore_errors=True)
print("MOONSHOT_FETCHED")
PYX
fi
echo "=== [4/6] 문샷 생성 (미확정분) ($(date +%H:%M:%S)) ==="
# **시간 상한을 건다 — D-Day 는 마감이 있는 작업이다** (2026-08-26 리허설로 신설).
# 종전에는 실패 시 `exit 1` 이라 **부분 생성물을 아예 못 썼다.** 문샷이 80% 끝난 상태로
# 죽으면 그 80% 가 통째로 버려지고 v2 단독으로 떨어진다. 마감이 있는 작업에서 이건 틀렸다.
# `DDAY_STAGE2_MAX_MIN` 을 주면 그만큼만 돌리고 **있는 만큼으로 진행**한다(fail-closed).
STAGE2_OK=1
if [ -n "${DDAY_STAGE2_MAX_MIN:-}" ]; then
  echo "[fail-closed] 2단계 시간 상한 ${DDAY_STAGE2_MAX_MIN}분"
  timeout "${DDAY_STAGE2_MAX_MIN}m" $PY src/inference/generate.py --input ${OUT}/stage2_input.csv \
    --output ${OUT}/moonshot_samp32.jsonl \
    --model /workspace/ckpt/moonshot_ep1 --n 32 --seed 42 --max-tokens 8192 \
    --chunk 100 --resume || STAGE2_OK=0
else
  $PY src/inference/generate.py --input ${OUT}/stage2_input.csv \
    --output ${OUT}/moonshot_samp32.jsonl \
    --model /workspace/ckpt/moonshot_ep1 --n 32 --seed 42 --max-tokens 8192 \
    --chunk 100 --resume || STAGE2_OK=0
fi
if [ "$STAGE2_OK" = "0" ]; then
  # **여기서 죽지 않는다.** 확보된 문항만으로 3단계·조립을 진행하고,
  # 문샷이 없는 문항은 **v2 단독 답으로 채워진다**(build_cascade 가 빈 묶음을 그렇게 다룬다).
  HAVE=$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1])}))" \
        "${OUT}/moonshot_samp32.jsonl" 2>/dev/null || echo 0)
  WANT=$($PY -c "import pandas,sys;print(len(pandas.read_csv(sys.argv[1])))" "${OUT}/stage2_input.csv")
  echo "**STAGE2_PARTIAL** — 문샷 ${HAVE}/${WANT} 확보. 나머지는 v2 단독으로 fail-closed."
  export DDAY_PARTIAL=1
fi
# 스크립트의 나머지가 --allow-partial 을 쓸지 결정한다
PARTIAL_FLAGS=""
[ "${DDAY_PARTIAL:-0}" = "1" ] && PARTIAL_FLAGS="--allow-partial-gens --allow-missing-partner"
ASM_PARTIAL=""
[ "${DDAY_PARTIAL:-0}" = "1" ] && ASM_PARTIAL="--allow-partial --allow-partial-gens"
T2=$(date +%s); echo "STEP2_SEC=$((T2-T1))"
backup_stage stage2

# 2단계까지의 제출물 — **전량 풀링이 아니다.** 1단계에서 확정된 문항은 문샷을 안 돌렸으므로
# 그 문항들은 v2 단독이다. 즉 이것은 "캐스케이드에서 3단계만 뺀 것"이지
# 전량 풀링(모든 문항에 v2+문샷 64표)과는 **다른 구성**이다.
# 전량 풀링으로 되돌리려면 문샷을 **전 문항에** 다시 돌려야 한다. (2026-08-25 외부 검토 지적)
$PY src/inference/make_submission.py \
  --gens ${OUT}/v2_samp32.jsonl ${OUT}/moonshot_samp32.jsonl \
  --k 32 32 --restrict-to-expect \
  --output ${OUT}/sub_2_cascade_no3.csv --expect-ids ${IN} || echo "SUB2_FAILED(비치명)"

# ---------- 3단계: 경시STaR (접전분만) ----------
echo "=== [5/6] 3단계 대상 선별 + 경시STaR 생성 ($(date +%H:%M:%S)) ==="
$PY src/eval/cascade_select.py --stage 3 --input ${IN} \
  --v2 ${OUT}/v2_samp32.jsonl --partner ${OUT}/moonshot_samp32.jsonl \
  --confirm ${CONFIRM} --margin ${MARGIN} ${PARTIAL_FLAGS} \
  --out ${OUT}/stage3_input.csv || { echo "SELECT3_FAILED"; exit 1; }

if [ ! -s /workspace/ckpt/exp-035_contest/adapter_model.safetensors ]; then
  echo "경시STaR 어댑터 회수 중"
  $PY - <<'PYX'
from huggingface_hub import snapshot_download
import shutil, os
snapshot_download("ahnjun0/dlc-artifacts", allow_patterns=["exp-035-contest-lora/*"],
                  local_dir="/workspace/dl035", max_workers=4)
os.makedirs("/workspace/ckpt/exp-035_contest", exist_ok=True)
src = "/workspace/dl035/exp-035-contest-lora"
for f in os.listdir(src):
    shutil.move(os.path.join(src, f), "/workspace/ckpt/exp-035_contest/")
shutil.rmtree("/workspace/dl035", ignore_errors=True)
print("CONTEST_FETCHED")
PYX
fi
$PY src/inference/generate.py --input ${OUT}/stage3_input.csv \
  --output ${OUT}/contest_samp32.jsonl \
  --lora /workspace/ckpt/exp-035_contest --n 32 --seed 42 \
  --chunk 100 --resume || { echo "**STAGE3_PARTIAL** — 확보분만으로 조립한다"; \
      export DDAY_PARTIAL=1; ASM_PARTIAL="--allow-partial --allow-partial-gens"; }
T3=$(date +%s); echo "STEP3_SEC=$((T3-T2))"
backup_stage stage3

# ---------- 최종 조립 ----------
echo "=== [6/6] 캐스케이드 조립 ($(date +%H:%M:%S)) ==="
$PY src/eval/build_cascade.py \
  --v2 ${OUT}/v2_samp32.jsonl \
  --partner ${OUT}/moonshot_samp32.jsonl \
  --third ${OUT}/contest_samp32.jsonl \
  --input ${IN} --confirm ${CONFIRM} --margin ${MARGIN} ${ASM_PARTIAL} \
  --out ${OUT}/sub_3_cascade.csv || { echo "ASSEMBLE_FAILED"; exit 1; }
T4=$(date +%s)

echo "=========================================="
echo "DDAY_TOTAL_SEC=$((T4-T0))  ($(( (T4-T0)/60 ))분)"
echo "  1) v2 전량        $(( (T1-T0)/60 ))분"
echo "  2) 문샷 미확정분  $(( (T2-T1)/60 ))분"
echo "  3) 경시 접전분    $(( (T3-T2)/60 ))분"
# **$PY 를 쓴다** — 앞에서 인터프리터를 탐지해놓고 여기서 `python` 을 직접 부르면
# PATH 에 python 이 없는 호스트에서 **모든 생성이 끝난 뒤 마지막에** 죽는다 (2026-08-25 외부 지적).
# 그리고 이 검증 실패는 **치명**으로 다룬다 — 행 수가 안 맞으면 제출하면 안 된다.
EXPECT=$("$PY" -c "import pandas,sys;print(len(pandas.read_csv(sys.argv[1])))" "${IN}") || { echo "VERIFY_FAILED(입력)"; exit 1; }
for f in ${OUT}/sub_*.csv; do
  N=$("$PY" -c "import pandas,sys;print(len(pandas.read_csv(sys.argv[1])))" "$f") || { echo "VERIFY_FAILED($f)"; exit 1; }
  echo "  ${N}행  $f"
  [ "$N" = "$EXPECT" ] || { echo "**행 수 불일치: $f 가 ${N}행, 입력은 ${EXPECT}행 — 제출 금지**"; exit 1; }
done
df -h /workspace | tail -1
backup_stage final
echo "**최종 제출본: ${OUT}/sub_3_cascade.csv**"
echo DDAY_CASCADE_DONE

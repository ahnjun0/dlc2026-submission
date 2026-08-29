#!/bin/bash
# **D-Day 자동 재실행 감시자** — 컨테이너 재시작·크래시 후 파이프라인을 스스로 되살린다.
#
# 왜 (2026-08-17 실측)
#   호스트 컨테이너가 1시간 반에 **2번 재시작**했다. 파이프라인에 `--resume` 은 있지만
#   `|| exit 1` 이라 **죽은 뒤 아무도 다시 켜주지 않는다.** 새벽 6시간 40분 런에서
#   한 번만 나도 통째로 날린다. 오늘은 4분 만에 사람이 알아챘지만 D-Day 엔 자고 있다.
#
# 설계
#   · cron 이 5분마다 이 스크립트를 부른다 (ssh 세션 독립 — 세션 종속 데몬은 반복 사망 실측)
#   · **비차단 flock**: 이미 돌고 있으면 즉시 종료 → 이중 실행 불가
#   · 완료 마커 파일이 있으면 즉시 종료 → 완료 후 재실행 안 함
#   · 죽었으면 락이 비어 있으므로 획득 성공 → `--resume` 으로 이어서 재개
#   · 자기 행동을 전부 로그에 남긴다 (조용한 실패 금지)
#
# 사용
#   설치:  crontab -l 2>/dev/null | grep -q dday_guard || \
#          (crontab -l 2>/dev/null; echo "*/5 * * * * bash /workspace/dlc/scripts/dday_guard.sh") | crontab -
#   해제:  touch /workspace/.dday_done   (또는 crontab -r)
# **출력 폴더는 여기서 고정한다** — 원격에서 sed 로 끼워 넣으면 다음 rsync 에 지워진다
# (8/17 실측: 원격 전용 편집이 scp 로 덮어써져 감시자가 1회차 폴더를 재개했다)
export REHEARSAL_OUT="${REHEARSAL_OUT:-experiments/exp-082_rehearsal2}"
# **경로를 환경변수로** (2026-08-28) — 재현 심사자가 자기 머신에서 돌릴 수 있어야 한다.
# 기본값은 종전 값 그대로다.
DLC_ROOT=${DLC_ROOT:-/workspace/dlc}
DLC_TMP=${DLC_TMP:-/workspace}
TARGET="${DDAY_TARGET:-$DLC_ROOT/scripts/run_dday_rehearsal.sh}"
DONE_MARK="${DDAY_DONE:-$DLC_TMP/.dday_done}"
LOG=$DLC_ROOT/logs/dday_guard.log
RUNLOG=$DLC_ROOT/logs/dday_run.log
mkdir -p "$DLC_ROOT/logs"
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

[ -f "$DONE_MARK" ] && exit 0

exec 8>"$DLC_TMP/.dday_guard.lock"
if ! flock -n 8; then exit 0; fi          # 이미 실행 중 → 조용히 종료

# **완료 판정은 산출물로 한다 — 로그 문자열로 하지 않는다** (8/17 실측 버그)
#   종전에는 누적 로그의 "REHEARSAL_DONE" 을 찾았는데, **한 번 찍히면 영구히 종료**됐다.
#   잘못된 대상이 완주하거나 부분 실행이 마커를 남기면 감시자가 다시는 안 뜬다.
#   → **제출 CSV 의 행 수가 입력 행 수와 일치하는지**로 판정한다. 이것이 진짜 완료 조건이다.
IN_CSV="${DDAY_INPUT:-$DLC_ROOT/data/raw/deep_chal_math_leaderboard_filtered.csv}"
# **제출물 파일명도 파라미터다** — 감시 대상이 `run_dday_cascade.sh` 면
# 최종본이 `sub_3_cascade.csv` 다. 박아두면 감시자가 영영 완료를 못 본다 (2026-08-26).
SUB="$DLC_ROOT/${REHEARSAL_OUT}/${DDAY_SUB:-rehearsal_submission.csv}"
if [ -s "$SUB" ] && [ -s "$IN_CSV" ]; then
  # ⚠ `wc -l` 은 못 쓴다 — 문제 본문에 따옴표 안 줄바꿈이 있어 831문항 CSV 가 1141줄이다 (8/17 실측)
  want=$(/venv/main/bin/python -c "import pandas,sys;print(len(pandas.read_csv(sys.argv[1]))+1)" "$IN_CSV" 2>/dev/null)
  got=$(( $(wc -l < "$SUB") ))          # 제출 CSV 는 `ID,answer` 정수뿐이라 줄바꿈 없음
  [ -z "$want" ] && { say "입력 행수 계산 실패 — 완료 판정 보류"; want=-1; }
  if [ "$got" -eq "$want" ]; then
    say "산출물 검증 통과 ($SUB : $got 행 = 입력 $want 행) → 감시 종료"
    touch "$DONE_MARK"; exit 0
  fi
  say "산출물 불완전 ($got / $want 행) — 계속 진행"
fi

# **고아 VLLM::EngineCore 회수** (8/17 실측 — 부모를 죽여도 자식이 VRAM 46.8GB 를 붙잡는다)
# 이걸 안 지우면 재시작해도 "Free memory 2.26/47.37 GiB" 로 초기화에 실패한다.
# ppid=1 인 것만 죽인다 — 정상 실행 중인 자식은 부모가 살아 있으므로 건드리지 않는다.
for gp in $(ps -eo pid,ppid,cmd | awk '$3=="VLLM::EngineCore" && $2==1 {print $1}'); do
  say "고아 EngineCore $gp 회수 (VRAM 점유 해제)"
  kill "$gp" 2>/dev/null; sleep 10
done

N=$(grep -c "^GUARD_LAUNCH" "$LOG" 2>/dev/null || echo 0)
say "GUARD_LAUNCH #$((N+1)) — 실행 중이 아님을 확인, --resume 으로 재개"
echo "GUARD_LAUNCH $(date -Iseconds)" >> "$LOG"
bash "$TARGET" >> "$RUNLOG" 2>&1
RC=$?
# 완료 마커도 대상마다 다르다. **다만 이건 보조 판정일 뿐** — 진짜 판정은 위의 산출물 행 수다
if grep -qE "${DDAY_MARK:-REHEARSAL_DONE}" "$RUNLOG" 2>/dev/null; then
  say "완주 (rc=$RC) → 감시 종료"; touch "$DONE_MARK"
else
  say "중단 (rc=$RC) — 5분 뒤 cron 이 재시도한다"
fi

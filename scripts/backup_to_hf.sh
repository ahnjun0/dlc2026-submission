#!/bin/bash
# 서버 산출물 → HF private (ahnjun0/dlc-artifacts) 백업
#
# **왜 습관화하는가** (2026-08-17)
#   호스트가 1시간 반에 두 번 재부팅했다. /workspace 는 비영속이고 인스턴스는 소멸할 수 있다.
#   오늘 롤아웃 하나가 **GPU 76분**이었다 — 재생성 비용이 백업 비용보다 훨씬 크다.
#
# **규칙**
#   ① 비싼 중간 생성물(생성 30분 이상)은 만든 즉시 백업
#   ② 판정에 쓰인 gens.jsonl 은 결과와 함께 백업 (재현 근거)
#   ③ 성능이 확인된 어댑터는 로컬 rsync + HF 이중
#   ④ **백업 성공을 확인하기 전에는 원본을 지우지 않는다**
#
# 사용:  bash scripts/backup_to_hf.sh <서버경로> <HF경로>
#   예:  bash scripts/backup_to_hf.sh experiments/exp-079_stepdpo exp-079-stepdpo
set -e
SRC="$1"; DST="$2"
[ -z "$SRC" ] || [ -z "$DST" ] && { echo "사용: $0 <서버상대경로> <HF경로>"; exit 1; }
# **호스트를 박아두지 말 것** (8/22 실측: 반납된 모스크바 서버 주소가 남아 백업이 조용히 실패했다)
# 새 인스턴스로 갈아탈 때는 `DLC_SSH` 만 바꾸거나 export 한다.
# **DLC_SSH 는 `sync_to_gpu.sh` 와 같은 규약 — ssh 인자만** (예: "-p 11378 root@<HOST>").
# 종전에는 이 스크립트만 "ssh ..." 전체 명령을 요구해, 같은 값을 두 스크립트에 못 쓰고
# `-p: command not found` 로 조용히 실패했다 (2026-08-25 실측). 둘 다 받아준다.
: "${DLC_SSH:?DLC_SSH 를 지정할 것 — 예: DLC_SSH='-p 11378 root@<HOST>'}"
case "$DLC_SSH" in ssh\ *) SSH="$DLC_SSH" ;; *) SSH="ssh -o ConnectTimeout=25 $DLC_SSH" ;; esac
echo "대상 호스트: $SSH"
$SSH "cd /workspace/dlc && /venv/main/bin/python - <<PY
import os, glob
from huggingface_hub import HfApi
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
api = HfApi(token=tok)
src, dst = '$SRC', '$DST'
files = [f for f in glob.glob(src + '/**/*', recursive=True) if os.path.isfile(f)]
tot = sum(os.path.getsize(f) for f in files)
print(f'{len(files)}개 파일 · {tot/1e9:.2f} GB 업로드')
api.upload_folder(folder_path=src, path_in_repo=dst, repo_id='ahnjun0/dlc-artifacts')
remote = [f for f in api.list_repo_files('ahnjun0/dlc-artifacts') if f.startswith(dst + '/')]
print(f'**검증: 원격 {len(remote)}개 파일 확인**')
assert len(remote) >= len(files), '업로드 누락 — 원본 삭제 금지'
print('OK')
PY"

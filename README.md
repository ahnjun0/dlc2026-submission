# 제5회 대학 연합 딥러닝 챌린지 2026 — 제출 저장소

**참가자** 안준영 (개인전) · **베이스 모델** `Qwen/Qwen2.5-3B-Instruct` (지정 모델, 변경 없음)
**과제** 수학 문제 → 정수 정답 (Exact Match)

---

## 1. 무엇을 제출하는가

최종 시스템은 **3단 적응형 캐스케이드**입니다. 세 모델 모두 지정 베이스에서 파생됐고,
**문항마다 표(vote)가 얼마나 모였는지에 따라** 다음 단계로 갈지를 결정합니다.

```
test 문항
  │
  ├─ ① STAR 32표 생성 ──────── 최다 득표 ≥ 22 ?
  │                             예  → 확정 (전 문항의 약 41%)
  │                             아니오
  │                               │
  │                               ├─ ② + LONGCOT 32표 (합 64표) ── 1·2위 격차 > 5 ?
  │                               │                                 예  → 확정 (약 46%)
  │                               │                                 아니오
  │                               │                                   │
  │                               │                                   └─ ③ + STAR-OLY 32표 (합 96표) → 확정 (약 13%)
```

**분기 변수는 「표 구조」뿐입니다.** 문제의 주제·출처·정답을 보지 않습니다.
따라서 유형별 라우팅이 아니며, 어떤 문항이 어느 단계로 갈지는 생성 결과만으로 정해집니다.

| 단계 | 이름 | 실험 ID | 형태 | 학습 데이터 | 역할 |
|---|---|---|---|---|---|
| ① | `STAR` | `exp-004_star_v2` | LoRA r128 | 대회 train 자기증류(STaR) | 주 모델 |
| ② | `LONGCOT` | `exp-015_moonshot` | full FT | R1 long-CoT 증류 | **다른 추론 체제**의 파트너 |
| ③ | `STAR-OLY` | `exp-035_contest` | LoRA r128 | 외부 경시 풀 자기증류 | 접전 문항 증원 |

**리더보드 0.80385 (668/831).** 단계 ③의 기여는 같은 생성 배치로 대조군을 재조립한
짝 검정으로 분리 확인했습니다(668 vs 665, `submissions/SUBMISSIONS.md` 참조).

---

## 2. 규칙 준수 — 검증 커맨드 포함

아래 커맨드는 이 저장소에서 그대로 실행하실 수 있고, 문서의 결과와 일치합니다.

### 2.1 베이스 모델은 `Qwen/Qwen2.5-3B-Instruct` 하나뿐입니다

```bash
git grep -hoE "Qwen/Qwen[0-9.]+-[A-Za-z0-9.-]+" -- '*.py' '*.sh' | sort | uniq -c
#   4 Qwen/Qwen2.5-3B-Instruct        ← 이것만 나옵니다
```

세 모델 전부 이 베이스에서 파생된 어댑터·미세조정이며 **가중치 병합(merge)은 없습니다.**
D-Day 스크립트가 불러오는 체크포인트도 셋뿐입니다.

```bash
git grep -oE "exp-[0-9]+_[a-z0-9_]+|moonshot_ep1" scripts/run_dday_cascade.sh | sort -u
#   exp-004_star_v2 / exp-035_contest / moonshot_ep1
```

### 2.2 추론 시 외부 호출이 없습니다

```bash
git grep -nE "requests\.|httpx|openai|urllib|api_key|https?://" -- src/inference/ src/eval/
#   검출 0건
```

추론은 **로컬 vLLM** 만 씁니다(`src/inference/generate.py`). 집계와 제출물 생성은
순수 파이썬 문자열 처리입니다(`src/eval/{parse,score,build_cascade}.py`).
코드 실행·도구 호출·계산기·외부 모델을 사용하지 않습니다.

> 참고: `src/data/match_train_sources.py` 에 `openai/gsm8k` 라는 문자열이 있는데,
> 이는 **HuggingFace 데이터셋 이름**(조직명이 openai)이며 API 호출이 아닙니다.
> 학습 데이터의 출처를 대조하는 오프라인 분석 도구입니다.

### 2.3 test·리더보드 문항을 학습에 쓰지 않았습니다

- 홀드아웃(`val_split`)은 어떤 학습에도 포함하지 않았습니다.
- 외부 데이터는 리더보드 문항과 **정규화 일치 기계적 제거**를 거쳤습니다.
- 리더보드 문항의 정답을 외부 데이터셋에서 조회하지 않았습니다.

### 2.4 사용한 외부 데이터

[`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md) 에 출처·라이선스·용도를
기록했습니다. 전부 **무료로 동등하게 접근 가능한 공개 데이터**입니다(NuminaMath-1.5,
OpenR1-Math, Omni-MATH, GSM-Plus, orca-math 등).

---

## 3. 환경

```bash
# GPU (학습·추론)
pip install -r requirements-gpu.txt

# 로컬 (평가·분석만, GPU 불필요)
pip install -r requirements-local.txt
pytest tests/ -q        # 72 passed
```

| 항목 | 값 |
|---|---|
| Python | 3.12 |
| torch | 2.11.0 **+cu130** |
| vllm | 0.26.0 |
| transformers / trl / peft | 5.14.1 / 1.9.2 / 0.20.0 |
| GPU | VRAM **48GB 권장** · CUDA 드라이버 **13.0 이상** |

> ⚠ **드라이버가 CUDA 13.0 미만이면 `torch.cuda.is_available()` 이 True 여도 커널에서 실패합니다.**
> Blackwell(SM 12.x)에서는 FlashInfer 샘플러를 꺼야 하며, `scripts/lib/vllm_env.sh` 가 자동 감지합니다.

---

## 4. 실행

### 4.1 가중치 받기

학습된 가중치 3종은 공개 저장소에 있습니다.

```bash
huggingface-cli download ahnjun0/dlc2026-weights --local-dir ./ckpt
# exp-004-v2-adapter/     STAR       LoRA    0.49 GB
# exp-015-moonshot/ep1/   LONGCOT    full FT 6.18 GB
# exp-035-contest-lora/   STAR-OLY   LoRA    0.98 GB
```

`scripts/run_dday_cascade.sh` 는 체크포인트가 없으면 **자동으로 받아옵니다.**

### 4.2 추론 실행

```bash
DLC_ROOT=$(pwd) \
DLC_CKPT=$(pwd)/ckpt \
DLC_TMP=/tmp \
DDAY_IN=path/to/test.csv \
DDAY_OUT=out \
bash scripts/run_dday_cascade.sh
```

**경로는 전부 환경변수로 지정합니다.** 하드코딩된 작업 디렉터리에 의존하지 않습니다.

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `DLC_ROOT` | `/workspace/dlc` | 이 저장소를 체크아웃한 경로 |
| `DLC_CKPT` | `/workspace/ckpt` | 가중치를 놓을 경로 |
| `DLC_TMP` | `/workspace` | 다운로드 스테이징 |
| `DLC_WEIGHTS_REPO` | `ahnjun0/dlc2026-weights` | 가중치 저장소 |
| `DDAY_IN` / `DDAY_OUT` | LB CSV / `experiments/dday_cascade` | 입력 문항 / 출력 경로 |
| `DDAY_CONFIRM` | `22` | 1단계 확정 문턱 |
| `DDAY_MARGIN` | `5` | 3단계 진입 격차 문턱 |

**산출물 3종**이 나옵니다. 앞 단계가 끝나는 즉시 하나씩 생기므로 중간에 중단돼도 제출물이 있습니다.

```
sub_1_v2only.csv        ① 직후 — STAR 단독 (보험)
sub_2_cascade_no3.csv   ② 직후 — 3단계 없는 캐스케이드
sub_3_cascade.csv       ③ 직후 — **최종 제출물**
```

### 4.3 처음부터 학습

[`REPRODUCE.md`](REPRODUCE.md) 가 데이터 생성 → 학습(3모델) → 추론 → 제출물 생성까지
실행 가능한 커맨드로 정리돼 있습니다. 커맨드는 실제로 실행해 검증했습니다.

```
§2.1  대회 데이터 · 홀드아웃 분리
§2.2  STaR 자기증류 데이터 (STAR)
§2.3  long-CoT 파트너 데이터 (LONGCOT)
§2.4  경시 풀 구축 + frontier 재확인 (STAR-OLY)
§3.1  STAR 학습    · §3.2 LONGCOT 학습 · §3.3 STAR-OLY 학습
§4    추론 · 캐스케이드 조립
§7    재현 시 알려진 함정 (전부 실측으로 겪은 것)
```

> **`LONGCOT` 은 `--epochs 2` 로 학습하고 1 epoch 체크포인트를 씁니다** — 정상적인 조기 종료
> 선택이며 `REPRODUCE.md §3.2` 에 명시돼 있습니다. 근거는 학습 곡선(2회차 loss 평탄화,
> epoch 경계의 계단 하락 = 암기 신호)이고, **ep2 와의 통제 비교는 하지 못했습니다.**

---

## 5. 소요 시간 (실측)

2,000문항 리허설을 **완주**해 잰 값입니다(2026-08-28, RTX PRO 5000 48GB).

| 단계 | 문항 | 시간 |
|---|---|---|
| ① STAR 32표 | 2,000 | 1h 45m |
| ② LONGCOT 32표 | 844 (42%) | 9h 54m |
| ③ STAR-OLY 32표 + 조립 | 약 360 (18%) | 30m |
| **합** | | **≈ 12h** |

`DDAY_CONFIRM=22` 는 **출력을 바꾸지 않는 계산 지름길**입니다 — 종전 `30` 과 비교해
LB 831 + val 460×3 + 리허설 2,000 = **4,211 사례에서 답 변경 0** 이며 2단계를 4시간 줄입니다
(`docs/…` 근거는 저장소 문서 참조). 되돌리려면 `DDAY_CONFIRM=30`.

---

## 6. 저장소 구성

```
src/inference/   generate.py          vLLM 생성 (로컬 전용)
                 make_submission.py   다수결 → 제출 CSV
src/eval/        parse.py             정수 파싱 (산술 없음 — 문자열 정규화만)
                 score.py             다수결 · 동률 정책
                 build_cascade.py     캐스케이드 조립
                 cascade_select.py    단계별 대상 선별
                 strict_verify.py     학습 전용 엄격 검증기 (제출 파서와 분리)
src/train/       sft_lora.py · sft_full.py
src/data/        make_split.py                 홀드아웃 분리
                 build_sft_v2.py               STaR 데이터
                 build_moonshot_data.py        long-CoT 데이터
                 build_contest_pool.py         경시 풀 정제
                 build_sft_contest.py          해결률 역비례 채택
scripts/         run_dday_cascade.sh   최종 실행
                 dday_guard.sh         cron 자동 재개 (컨테이너 재시작 대비)
                 provision_instance.sh 환경 구축
                 lib/vllm_env.sh       vLLM 환경 (Blackwell 자동 감지)
tests/           72개 — 파서 · 집계 · 환경 회귀
```

**주요 문서**

| 문서 | 내용 |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **재현 안내 — 심사 시 이 문서를 보시면 됩니다** |
| [`docs/methodology.md`](docs/methodology.md) | 방법론 · 판정 규칙 · 자기 평가 |
| [`experiments/RESULTS.md`](experiments/RESULTS.md) | 실험 대장 — 모든 실험의 변경점·결과·결론 |
| [`docs/axis-map.md`](docs/axis-map.md) | 축 지도 — 닫은 43개 축의 실측 근거 |
| [`submissions/SUBMISSIONS.md`](submissions/SUBMISSIONS.md) | 제출 이력과 LB 짝 검정 |
| [`docs/dday-protocol.md`](docs/dday-protocol.md) | 최종일 실행 절차 |

---

## 7. 방법론 요약

### 7.1 판정 규칙 (모든 실험에 사전 등록)

두 달간 **43개 축**을 열고 닫았습니다. 그 판정을 신뢰할 수 있게 만든 규칙이 방법론의 본체입니다.

| 관문 | 내용 | 도입 계기 |
|---|---|---|
| ① 문항 부트스트랩 | **P(이득) ≥ 95%** | val 460은 1%p 미만을 판정 못 한다 |
| ② 다중 시드 | 최소 2시드 · 기준선은 4시드 **평균** | 같은 모델의 시드 폭이 **1.96%p** |
| ③ 주 모델 시드 복제 | 승자를 다른 시드로 재현 | 하루에 네 번 후보를 걸러냈다 |
| ④ 문항 폴드 교차적합 | 시드 폴드 금지 | 누설로 +3.67이 실제 **−2.00** |
| ⑤ LB 짝 검정 | 같은 배치로 대조군 재조립 | val은 접전 지대 ±3이 잡음 |

**역예측 검증**: 새 판정법을 도입할 때 *이미 결과를 아는 실패 사례*를 먼저 넣어
기각되는지 확인한 뒤에만 신규 후보에 씁니다. 이 규칙은 이후 세 번 적중했습니다.

### 7.2 남기는 관측

`maj@k` 에서 **점수를 지배하는 것은 능력이 아니라 표의 분산**이었습니다.
능력이 비슷한 7개 모델(`pass@32` 90.87~92.61%)에서:

```
pass@32 ↔ maj@32   r = +0.134     ← 능력은 점수를 거의 예측 못 한다
n_eff   ↔ maj@32   r = −0.920     ← 분산이 점수를 결정한다
```

캐스케이드의 세 자리가 각각 다른 것을 요구하는 이유가 여기서 나옵니다 —
주 모델은 표가 모여야 하고, 2단계 파트너는 **다른 추론 체제**여야 하며
(오답 충돌률 24.7% vs 같은 모델 재뽑기 66.7%), 3단계는 **그 지대를 실제로 잘 푸는 모델**이어야
합니다(어려운 문항 +3.53%p / 중간 난이도 −3.51%p / 쉬운 문항 불변).

한계도 함께 적습니다 — 이 상관은 **n=7~9의 시사**이지 확정이 아니며,
좁은 구간(`n_eff` 2.77 대 3.03) 안에서는 예측력이 없습니다.

---

## 8. 라이선스와 저작

베이스 모델 `Qwen/Qwen2.5-3B-Instruct` 는 Apache-2.0,
사용한 외부 데이터의 라이선스는 [`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md)에
개별 명시했습니다. 이 저장소의 코드는 대회 제출·검증 목적으로 공개합니다.

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

## 1-A. 이름·경로 대응표 — 먼저 보시면 헷갈리지 않습니다

문서에서는 **역할 이름**(`STAR`·`LONGCOT`·`STAR-OLY`)으로 부르고,
코드·파일에서는 **개발 당시 식별자**를 그대로 씁니다. 대응은 다음과 같습니다.

| 역할 이름 | 실험 ID | 체크포인트 디렉터리 | HuggingFace 경로 | 조립 CLI 플래그 | 생성물 파일 |
|---|---|---|---|---|---|
| **STAR** | `exp-004` | `exp-004_star_v2` | `exp-004-v2-adapter/` | `--v2` | `v2_samp32.jsonl` |
| **LONGCOT** | `exp-015` | `moonshot_ep1` | `exp-015-moonshot/ep1/` | `--partner` | `moonshot_samp32.jsonl` |
| **STAR-OLY** | `exp-035` | `exp-035_contest` | `exp-035-contest-lora/` | `--third` | `contest_samp32.jsonl` |

> **식별자를 개명하지 않은 이유**: 커맨드·경로·산출물 이름은 코드와 **바이트 단위로 일치**해야
> 재현이 됩니다. 문서만 예쁘게 고치면 실행이 깨집니다. 대신 이 표로 연결합니다.
>
> 부록 문서(`experiments/RESULTS.md` · `docs/axis-map.md`)는 개발 기간의 표기(`v2` · `문샷` ·
> `경시STaR`)를 그대로 두었습니다 — 위 표의 **역할 이름 → 실험 ID** 로 대조하시면 됩니다.

---

## 1-B. 왜 이렇게 만들었는가 — 설계 근거

세 모델과 3단 분기는 임의로 고른 구성이 아닙니다. 두 달간 **43개 축**을 열고 닫으면서
얻은 관측 위에 세웠고, 근거 문서를 저장소에 함께 넣었습니다.

### 관측 1 — `maj@k` 에서 점수를 지배하는 것은 능력이 아니라 **표의 분산**이다

능력이 비슷한 7개 모델(`pass@32` 90.87~92.61%)에서:

```
pass@32 ↔ maj@32   r = +0.134     능력은 점수를 거의 예측하지 못한다
n_eff   ↔ maj@32   r = −0.920     분산(Kish 유효표수)이 점수를 결정한다
```

두 달간 우리가 올린 것은 **왼쪽 축**이었고, 그래서 학습 축이 계속 실패했습니다
(SFT 0승 10패 · 구조 0승 3패 · 목적함수 0승 3패). 남은 것은 **집계**였고 그것만 LB에서 실증됐습니다.
→ [`docs/dispersion-law-2026-08-27.md`](docs/dispersion-law-2026-08-27.md)

### 관측 2 — 다양성은 「데이터」가 아니라 **「추론 체제」**에서 온다

*둘 다 틀렸을 때 같은 오답을 내는가*를 재고, **같은 모델을 시드만 바꿔 다시 뽑은 것**을
「다양성 없음」 기준선으로 넣었습니다.

```
STAR ↔ LONGCOT           24.7%   ← 다른 체제 (기준선의 1/2.7)
STAR ↔ STAR-OLY          57.7%   ← 학습 데이터만 다름 (기준선과 사실상 같다)
STAR ↔ STAR (시드만)      66.7%   ← 대조군
```

`STAR` 와 `STAR-OLY` 는 **학습 데이터가 다른데도** 오답이 대조군만큼 겹칩니다.
반면 `LONGCOT` 만 다릅니다 — 표당 길이 10,425자, `wait` 17.05회로 R1 의 자문자답 화법을 물려받아
**추론 체제 자체가 다릅니다**(다른 둘은 1,412·1,430자 · `wait` 0.01).
→ [`docs/why-cascade-works-2026-08-26.md`](docs/why-cascade-works-2026-08-26.md)

### 관측 3 — 세 자리는 **각각 다른 것**을 요구한다

| 자리 | 질문 | 필요한 것 | 배치 근거 |
|---|---|---|---|
| ① 주 모델 | — | 표가 **모여야** 한다 | `n_eff` 낮을수록 점수 높다 (r −0.920) |
| ② 파트너 | 「이 답이 맞나?」 | **다른 시각** | 오답 충돌률 24.7% (대조군 66.7%) |
| ③ 증원 | 「64표가 갈렸다, 누가 맞나?」 | **더 나은 실력** | 어려운 문항 +3.53%p |

**그래서 순서가 「길이 사다리」가 아닙니다.** 직관적으로는
`STAR → STAR-OLY → LONGCOT`(1,152 → 1,175 → 6,106자)이 자연스러워 보이지만,
실측은 **−2.33문항**입니다. 다양성이 필요한 자리에 전문가를 놓고, 실력이 필요한 자리에
다양성을 놓기 때문입니다.

> **캐스케이드는 「모델이 점점 더 애쓰는」 사다리가 아니라 「우리가 점점 더 의심하는」 사다리입니다.**

### 관측 4 — 전역 개입은 **교환비 장벽**에 막힌다

`ΔAcc = (1−π)r − πb`. 확신 구간의 실측 기저율이 **98.2%** 이므로 이기려면
**고치는 비율 / 망치는 비율 > 54.6** 이 필요합니다. 우리가 잰 전역 개입은 전부 못 넘었습니다.
그런데 **접전 지대의 기저율은 23%** 입니다 — 위로 갈 여지가 있습니다.

**닫힌 축의 상당수는 기법이 나빠서가 아니라 「게이트가 없어서」 죽었습니다.**
③단계가 유일하게 이긴 이유가 이것이고, 그래서 이 시스템은 **조건부 개입**입니다.

### 3단계의 기제 — 「표 증량」이 아니다

같은 표 수(96)에서 3번째 자리만 바꿔봤습니다.

```
STAR-OLY  +3.00    >    STAR 재뽑기  +1.67    >    LONGCOT 증량  −2.00
```

전역 증량은 비율을 유지한 채 몬테카를로 분산만 줄입니다. ③단계는 접전 문항에서
**STAR 계열 : LONGCOT 계열 = 1:1 → 2:1** 로 바꿉니다. 접전 지대는 gold 점유 0.188 vs
최다 오답 0.243 이라, **같은 분포에서 표만 늘리면 틀린 모드로 더 안정적으로 수렴합니다.**

> 정확한 요약: **「접전 구간에서의 추가 표집이자 STAR 계열에 대한 조건부 재가중」.**

### 한계 (함께 밝힙니다)

- 위 상관은 **n = 7~9 의 시사**이지 확정이 아닙니다.
- **좁은 구간 안에서는 예측력이 없습니다** — `STAR-OLY` 의 `n_eff` 는 세 시드 모두
  `STAR` 보다 낮은데 `maj@32` 는 동률입니다(Δ 평균 +0.33문항 · 양수 시드 1/3).
- 홀드아웃(val 460)은 우리 라벨 정화 때문에 **낙관 편향**이 있습니다(−2.32%p).
  외부 1,169문항으로 보정 곡선을 만들어 확인했습니다.
  → [`docs/methodology.md`](docs/methodology.md)

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
git grep -hoE "exp-[0-9]+_[a-z0-9_]+|moonshot_ep1" scripts/run_dday_cascade.sh | sort -u
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
- 리더보드/test 문항의 정답을 외부 데이터셋에서 **조회하지 않았습니다.**

**출처 매칭에 대해 — 먼저 밝힙니다.** 대회 train 문항이 어느 공개 데이터셋에서 왔는지
식별하는 분석을 했습니다(`src/data/match_train_sources.py`). 목적은 **SFT 데이터의 도메인
정렬**이며, 규칙 5.2 가 허용하는 공개 데이터를 어떻게 고를지 정하기 위해서입니다.
train 은 정답이 함께 제공되므로 이 매칭으로 **새로 얻는 라벨이 없습니다.**

그 분석은 **train 만** 대상으로 하며 리더보드/test 파일을 읽지 않습니다.

```bash
grep -n "leaderboard\|test" src/data/match_train_sources.py
#   주석 한 줄만 나옵니다 — "leaderboard/test 문항은 분석에서 제외"
```

**그리고 감사 결과를 있는 그대로 밝힙니다 — 완전히 0건은 아니었습니다.**

8/29 에 정규화를 강화해(영숫자만 남김) 배포 3모델의 학습 데이터 68,731샘플 전체를
리더보드 **원본 1,000행**과 재대조했습니다.

| 학습 데이터 | 샘플 수 | LB 원본 1,000 일치 | **채점 대상 831 일치** |
|---|---|---|---|
| `sft_v2.jsonl` (STAR — **주 모델**) | 19,185 | 2 | **0건** |
| `moonshot_r1.jsonl` (LONGCOT) | 25,953 | 30 | **2건** |
| `sft_contest.jsonl` (STAR-OLY) | 23,593 | 20 | **1건** (LONGCOT 과 중복) |
| **합집합** | 68,731 | 52 | **2문항 = 0.24%** |

두 문항(`val-000575` · `val-000787`)은 실제로 같은 문제였고, 표현만 달랐습니다
(줄바꿈·LaTeX 감쌈). **이전 감사가 0건이었던 것은 없어서가 아니라 정규화가 약해서**
(공백만 축약) 그 차이를 못 잡았기 때문입니다. 같은 파일을 옛 정규화로 재실행하면
지금도 0건이 나옵니다 — 검사 방법의 한계였습니다.

**규칙 대조**: 5.1b 가 금지하는 것은 *"test.parquet 의 문제를 학습 데이터로 사용"* 이며,
리더보드 문항의 학습 사용을 금지한 조항은 없습니다. 10.1a 가 금지하는 *"정답의 외부 확보"*
에도 해당하지 않습니다 — 배포된 리더보드 파일에는 `answer` 컬럼 자체가 없고, 우리 제출 답은
전부 모델 추론 산출물입니다. **주 모델 STAR 은 채점 대상 일치가 0건**입니다.

원인은 원천이 같다는 점입니다. 우리 외부 데이터(`OpenR1-Math-220k` · `NuminaMath-1.5`)와
대회 데이터가 같은 계열이라 문항이 겹칩니다. 빌드 시점의 dedup 은 필터본 831행 + 약한
정규화를 썼고, **절차는 원본 1,000행까지 대조하도록 교정**했습니다
(`src/data/build_contest_pool.py` — 그래서 STAR-OLY 풀은 그 이후 산출물입니다).

> 전체 감사 결과와 나머지 회색 지대 3건(자체 holdout 21문항 · 운영진 오류 목록 627 ·
> 가중치 공개 시점)은 **[`docs/rules-compliance-audit.md`](docs/rules-compliance-audit.md)**
> 에 수치·영향과 함께 공개했습니다. 검증기 자체도 저장소에 있습니다
> (`src/eval/rules_audit.py` — 조항별 PASS/WARN/FAIL 을 재현하실 수 있습니다).

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

**문서 — 심사용과 부록을 구분했습니다**

| 심사용 | 내용 |
|---|---|
| **이 README** | 시스템·설계 근거·규칙 준수·실행법 |
| [`REPRODUCE.md`](REPRODUCE.md) | **재현 안내** — 데이터 생성 → 학습(3모델) → 추론 → 제출물 |
| [`docs/methodology.md`](docs/methodology.md) | **방법론 문서** (규칙 8.2) — 판정 규칙·실험 설계·자기 평가 |
| [`docs/rules-compliance-audit.md`](docs/rules-compliance-audit.md) | **규정 준수 자체 감사** — 조항별 검사 결과와 회색 지대 4건의 수치 공개 |
| [`docs/rules-verbatim-2026-08-29.md`](docs/rules-verbatim-2026-08-29.md) | 대조 근거가 된 규칙·공지 원문 발췌 |
| [`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md) | **사용 데이터 목록** (규칙 5.2c) |

| 부록 (연구 기록) | 내용 |
|---|---|
| [`experiments/RESULTS.md`](experiments/RESULTS.md) | 실험 대장 — 모든 실험의 변경점·결과·결론 |
| [`docs/axis-map.md`](docs/axis-map.md) | 축 지도 — 닫은 43개 축의 실측 근거 |
| [`docs/dispersion-law-2026-08-27.md`](docs/dispersion-law-2026-08-27.md) | 분산 법칙 (관측 1의 근거) |
| [`docs/why-cascade-works-2026-08-26.md`](docs/why-cascade-works-2026-08-26.md) | 캐스케이드 기제 (관측 2·3의 근거) |
| [`docs/irt-analysis-2026-08-16.md`](docs/irt-analysis-2026-08-16.md) | IRT 재판정 — 측정 도구 자체의 검토 |
| [`docs/stage3-ablation-preregistration.md`](docs/stage3-ablation-preregistration.md) | 사전 등록 예시 (3단계 ablation) |
| [`submissions/SUBMISSIONS.md`](submissions/SUBMISSIONS.md) | 제출 이력과 LB 짝 검정 |

> **부록은 개발 기간 중 유지한 작업 일지입니다.** 자기 앞으로 쓴 지침과 날짜별 기록이
> 그대로 남아 있고 결론이 뒤에서 정정된 곳도 있습니다. 위 심사용 문서의 주장을
> 뒷받침하는 **근거**로 함께 두며, 읽는 순서는 README → REPRODUCE 입니다.

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

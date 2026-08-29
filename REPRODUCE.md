# 재현 안내 (Reproduction Guide)

제5회 대학 연합 딥러닝 챌린지 · 참가자 **안준영** (개인전)

대회 규칙 **[8.2]** 이 요구하는 ① 학습 코드 ② 추론 코드 ③ 모델 가중치 ④ 데이터셋 목록
⑤ 환경 설명 ⑥ 방법론 문서를 이 문서와 링크된 문서들이 담는다.

- 방법론: [`docs/methodology.md`](docs/methodology.md)
- 실험 대장(전 실험 기록): [`experiments/RESULTS.md`](experiments/RESULTS.md)
- 축 지도(폐쇄 축과 근거): [`docs/axis-map.md`](docs/axis-map.md)
- 데이터 출처: [`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md)
- 제출 이력: [`submissions/SUBMISSIONS.md`](submissions/SUBMISSIONS.md)

---

## 0. 최종 제출 구성 (한눈에)

```
베이스 모델   Qwen/Qwen2.5-3B-Instruct        (규칙 4.1 — 유일한 출발점, 병합 없음)
주 모델       exp-004_star_v2   LoRA r128     자기증류(STaR) SFT
파트너        exp-015_moonshot  full FT       long-CoT 증류
3단계 증원    exp-035_contest   LoRA          경시 STaR (접전 문항에만)
집계          **적응형 캐스케이드**(exp-101) · 순서 무관 동률 정책 다수결
              ① v2 32표의 최다 득표 ≥30  →  v2 단독으로 확정
              ② 아니면 v2 + 문샷 = 64표 풀링
              ③ 그 1·2위 격차 ≤5  →  경시STaR 32표 증원 (96표)
```

**분기는 표 구조로만 정해진다** — gold 도 출처도 보지 않으므로 8/4 유권해석의
*"유형별 특화 + 라우팅"* 에 해당하지 않는다.

리더보드 실측 (LB 831 · 같은 생성 배치로 짝 고정):
```
적응형 캐스케이드 (채택)   **0.80385 = 668/831**   ← sub-014
전량 풀링 (대조군)         0.80024 = 665/831      ← sub-012 와 0행 차이로 재현
```
로컬 val460: 캐스케이드 83.26% · 3시드 짝 검정 [+6, +3, +0] · 문항 부트스트랩 P **98.8%**.
> ⚠ **정직한 단서**: LB 단독 McNemar 는 유의하지 않다(33행 변동에 순증 3).
> 채택 근거의 무게는 val 다시드·교차적합에 있고, LB 는 **전이 확인**이다.
> (`sub-008` 0.80385/668 은 exp-055 를 포함한 **다른 3자 구성**이며 그 축은 P=46.3% 로 폐쇄됐다.)

**3단계의 독립 검증 (2026-08-27 추가)**: 위 «캐스케이드 vs 전량 풀링» 비교는 1·2·3단계
효과가 섞여 있어 **3단계 자체의 근거가 아니었다.** 조립만 바꾼 순수 ablation 으로 재검정했다:
```text
sub-014  3단계 있음   0.80385 = **668**/831
sub-017  3단계 없음   0.80024 = **665**/831      **Δ_LB = +3문항** (갈린 행 33)
```
사전 등록표(`docs/stage3-ablation-preregistration.md`)의 «Δ ≥ +3 → 유지» 에 해당하며,
val 3시드(382.00 → 385.00 = +3.00)와 **방향·크기가 일치**한다.

**3단계의 기제** (`docs/why-cascade-works-2026-08-26.md` §8): 추가 표 수를 32로 고정하고
구성만 바꾸면 **경시STaR +3.00 > v2 재뽑기 +2.00 > 문샷 −2.00** 이다.
즉 3단계는 «표 증량» 이 아니라 **«접전 구간에서 v2 계열로의 조건부 재가중»** 이다.
2단계는 «다른 **체제**»(문샷)를, 3단계는 «같은 체제 · 다른 **데이터**»(경시STaR)를 요구한다.

**추론 시 외부 모델·인터넷·코드 실행을 일절 사용하지 않는다** (규칙 4.3 / 6.a).
집계는 다수결이며 이는 규칙 **[6.c]** 가 명시적으로 허용한다.

### 소요 시간과 fallback (2,000문항 실측 기준)

```text
표준 구성   confirm 30 · margin 5           **약 12.3시간**  (PRO 5000 급)
            1단계 v2 2,000문항 1시간 45분 · 2단계 문샷 0.68분/문항 · 3단계 ~35분

**측정된 시간 fallback**
  `--tier1-n 16 --tier1-confirm 12 --confirm 16`  →  **7~8시간**
  비용 **LB −2문항** (sub-016 666 vs sub-014 668) — **5시간을 2문항에 산다**
  발동 조건 ① test 2단계 진입률 급증 ② 장비 미달 ③ 시작 지연
```
**단계마다 제출물이 확보되고 자동 HF 백업된다** — `sub_1_v2only` → `sub_2_cascade_no3`
→ `sub_3_cascade`. 어느 단계가 죽어도 앞 단계 답으로 **fail-closed** 되며 빈칸이 없다.

---

## 1. 환경

```
GPU        NVIDIA RTX 4090 48GB (개조판) · Vast.ai
호스트     CUDA 13.2 (드라이버 ≥ 13.0 **필수** — 12.x 는 is_available()=True 여도 커널 실패)
Python     3.12 · /venv/main
핵심 패키지 torch 2.11+cu130 · transformers · trl · peft · vllm
정밀도     bf16 (Ampere 이상 필수)
작업 경로   /workspace/dlc
```

패키지 목록: [`requirements-gpu.txt`](requirements-gpu.txt)

**결정성**: 모든 판정용 추론은 `VLLM_BATCH_INVARIANT=1` · `VLLM_ENABLE_V1_MULTIPROCESSING=0`
로 실행했다. 이것만으로 **실행 간 잡음이 0** 이 된다. 단 **청크 크기가 다르면 여전히 갈리므로**
(일치율 99.22%) A/B 비교는 `--chunk` 까지 고정해야 한다.

---

## 2. 데이터

### 2.1 대회 데이터

```
data/raw/train.csv                                   17,000문항 (원본, 수정 금지)
data/raw/train_filtered_ids.csv                      공식 오류 627건 (8/3 공개)
data/raw/deep_chal_math_leaderboard_filtered.csv     리더보드 831문항
data/processed/train_split.csv / val_split.csv       16,500 / 500 (seed 42 층화 분할)
data/processed/val_split_corrected.csv               **460문항 — 모든 평가의 기준**
```

`val_split_corrected.csv` 는 500문항에서 공식 제외 27 + 모호·훼손 4 + 결함 7 +
고난도 라벨오류 2 를 뺀 **460문항**이며 확정 14건을 재라벨했다. LB 채점(제외+교정)과 정합한다.

```bash
python src/data/make_split.py            # train/val 분할 (seed 42)
```

### 2.2 학습 데이터 생성 (STaR 자기증류)

**핵심**: 외부 해설을 쓰지 않고 **베이스 모델 자신의 정답 풀이만** 학습에 쓴다.

```bash
# ① 자기 생성 (train_split 에 대해 n=8 샘플링)
python src/inference/generate.py \
  --input data/processed/train_split.csv \
  --output experiments/exp-004_star/train_samp8.jsonl \
  --n 8 --seed 42 --chunk 250 --resume

# ② 정답 도달분만 채택 → SFT 데이터
python src/data/build_sft_v2.py          # → data/processed/sft_v2.jsonl (19,185건)
```

**리더보드 문항 제거(dedup)** 가 빌더에 내장되어 있다 — 정규화 일치 항목을 기계적으로 제외한다
(규칙 5.1b / 5.3b). test 문항의 정답을 외부에서 조회하는 행위는 **일절 하지 않았다**.

### 2.3 long-CoT 파트너 데이터

```
출처: **OpenR1-Math-220k** (`open-r1/OpenR1-Math-220k`, Apache-2.0)
93,733건 → 필터 후 **25,953건**

python src/data/build_moonshot_data.py   # → data/processed/moonshot_r1.jsonl
```

---

### 2.4 3단계 멤버 데이터 (경시 STaR)

배포 구성의 세 번째 모델(`exp-035_contest`)이 쓰는 데이터다.
**대회 train 이 아니라 외부 경시 풀**을 쓰며, 주제가 아니라 **해결률(난이도)** 로 고른다.

```bash
# ① 경시 계열 정제 풀 (NuminaMath-1.5 의 품질 플래그 활용 — MCQ·미해결 해설 제외)
python src/data/build_contest_pool.py --output data/processed/contest_pool.csv

# ② 자기 생성 (주 모델로 n=8) — 이것이 STaR 의 자기증류 부분
python src/inference/generate.py --input data/processed/contest_pool.csv \
  --output experiments/exp-035_contest_star/contest_samp8_s42.jsonl \
  --lora /workspace/ckpt/exp-004_star_v2 --n 8 --seed 42 --chunk 500 --resume

# ③ frontier(8샘플 중 1개만 정답) 항목은 **독립 시드로 재확인**한다
#    — 틀린 풀이가 우연히 정답 정수에 착지했을 수 있어서다
python src/inference/generate.py --input data/processed/contest_frontier.csv \
  --output experiments/exp-035_contest_star/frontier_samp8_s43.jsonl \
  --lora /workspace/ckpt/exp-004_star_v2 --n 8 --seed 43 --chunk 500 --resume

# ④ 해결률 역비례 채택 (easy 1배 / mid 2배 / frontier 4배) → SFT 데이터
python src/data/build_sft_contest.py \
  --gens experiments/exp-035_contest_star/contest_samp8_s42.jsonl \
  --recheck-gens experiments/exp-035_contest_star/frontier_samp8_s43.jsonl \
  --output data/processed/sft_contest.jsonl
```

> **v2 데이터와 혼합하지 않는다.** 1:4 로 희석했더니 φ 가 0.726 → 0.878 로 올라
> 파트너 가치가 절반이 됐다(실측). 희석은 다양성을 죽인다.

## 3. 학습 재현

### 3.1 주 모델 — `exp-004_star_v2` (LoRA)

```bash
python src/train/sft_lora.py \
  --data data/processed/sft_v2.jsonl \
  --output /workspace/ckpt/exp-004_star_v2 \
  --lr 2e-5 --epochs 2 --seed 42
```

체크포인트에 보존된 실제 인자 (`training_args.bin`):

```
learning_rate 2e-05 · num_train_epochs 2.0 · seed 42
per_device_train_batch_size 1 · gradient_accumulation_steps 32 · bf16 True
LoRA: r=128 · alpha=256 · dropout=0.0 · target=all-linear 7종
      (q,k,v,o,gate,up,down_proj) · base=Qwen/Qwen2.5-3B-Instruct
```

> **⚠ 재현 시 주의 — `--attn` 을 지정하지 않는다.**
> 챔피언은 기본 attention 으로 학습했다. `--attn kernels-community/flash-attn2` 를 주면
> `exp-022_clean_v2` 가 되며 **다른 모델**이다 (val 81.74 vs 82.17, 단 P=10.4% 로 통계적 동률).
> 후속 학습의 표준은 clean 설정이지만 **제출 모델은 기본 설정 산출물**이다.

### 3.2 파트너 — `exp-015_moonshot` (full FT)

```bash
python src/train/sft_full.py \
  --data data/processed/moonshot_r1.jsonl \
  --output /workspace/ckpt/exp-015_moonshot \
  --epochs 2 --save-steps 100 --seed 42
```

**제출에 쓰는 것은 1 epoch 체크포인트**(`ep1`)다. 조기 종료(early stopping)에 해당하며,
`--epochs 2` 로 돌린 뒤 그 지점을 채택한다.

**고른 근거**: 학습 곡선이다. 2회차 안에서 loss 가 0.4333 → 0.4266 (1.5%)로 이미 평평하고,
epoch 경계의 계단 하락(0.4649 → 0.4333)은 암기 신호로 읽었다.
**val 점수나 리더보드 점수로 고르지 않았다.**

> ⚠ **정직하게 적는다 — `ep2` 와의 통제 비교는 하지 못했다.**
> 2026-08-08 에 「추가 epoch = 파트너 가치 하락」이라고 적었다가 **LR 스케줄 교락으로 철회**했다
> (cosine 2-epoch 스케줄의 1 epoch 지점은 LR 이 피크의 52.4% 로, 완전히 어닐링된 1-epoch 런과 다르다).
> 따라서 **`ep2` 가 파트너로 더 나은지 우리는 모른다.** 배포 구성이 `ep1` 인 이유는
> 「더 낫다고 측정해서」가 아니라 **「그 구성이 리더보드에서 실증됐기 때문」**이다.

---

### 3.3 3단계 멤버 — `exp-035_contest` (LoRA)

**레시피는 주 모델과 완전히 동일하게 고정**한다 — LoRA r128 all-linear · LR 2e-5 ·
2 epoch · seq 4096 · flash-attn2. **유일한 변수는 문제 풀 구성**이다.

```bash
python src/train/sft_lora.py \
  --data data/processed/sft_contest.jsonl \
  --output /workspace/ckpt/exp-035_contest \
  --lr 2e-5 --epochs 2 --seed 42 \
  --attn kernels-community/flash-attn2

# 종료코드가 아니라 **산출물로 판정한다**
find /workspace/ckpt/exp-035_contest -name adapter_model.safetensors
```

전 과정을 한 번에 돌리려면 `scripts/run_contest_pipeline.sh` 를 쓴다.

## 4. 추론 재현 (제출물 생성)

```bash
export VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# ⚠ **Blackwell(sm120) 계열에서는 아래가 추가로 필요합니다** (2026-08-24 실측)
#    RTX PRO 5000/6000 · B200 · 5090 등에서 vLLM 0.26 의 FlashInfer 샘플러가
#    "requires GPUs with sm75 or higher" 로 죽습니다. 실제 원인은 능력 탐지 실패
#    ("Failed to get device capability: SM 12.x requires CUDA >= 12.9") 이며,
#    sm120 은 sm75 보다 **높은데도** 가드에 걸립니다.
#    (VLLM_ATTENTION_BACKEND 는 vLLM 0.26 이 더 이상 읽지 않아 무효입니다.)
export VLLM_USE_FLASHINFER_SAMPLER=0   # Blackwell 에서만

# ① 주 모델 32표
python src/inference/generate.py --input <TEST.csv> \
  --output out/v2_samp32.jsonl \
  --lora /workspace/ckpt/exp-004_star_v2 \
  --n 32 --seed 42 --chunk 100 --resume

# ② 파트너 32표 (long-CoT 이므로 max-tokens 8192)
python src/inference/generate.py --input <TEST.csv> \
  --output out/moonshot_samp32.jsonl \
  --model /workspace/ckpt/moonshot_ep1 \
  --n 32 --seed 42 --max-tokens 8192 --chunk 100 --resume

# ③ 3단계 대상 선별 → 경시STaR 증원
python src/eval/cascade_select.py --stage 3 --input <TEST.csv> \
  --v2 out/v2_samp32.jsonl --partner out/moonshot_samp32.jsonl \
  --out out/stage3_input.csv
python src/inference/generate.py --input out/stage3_input.csv \
  --output out/contest_samp32.jsonl \
  --lora /workspace/ckpt/exp-035_contest \
  --n 32 --seed 42 --chunk 100 --resume

# ④ 캐스케이드 조립 → 제출 CSV (첫 인자가 주 모델 = 동률 우선권)
python src/eval/build_cascade.py \
  --v2 out/v2_samp32.jsonl --partner out/moonshot_samp32.jsonl \
  --third out/contest_samp32.jsonl \
  --input <TEST.csv> --confirm 30 --margin 5 --out submission.csv
```
> ②는 **전량이 아니라 ①에서 확정되지 않은 문항만** 돌린다
> (`cascade_select.py --stage 2` 로 입력 CSV 를 만든다). LB 실측 분기: ① 462 / ② 369 / ③ 107.

전 과정 자동화: [`scripts/run_dday_cascade.sh`](scripts/run_dday_cascade.sh)
신규 인스턴스 가동: [`scripts/provision_instance.sh`](scripts/provision_instance.sh)
장애 자동 복구: [`scripts/dday_guard.sh`](scripts/dday_guard.sh) (cron 5분 주기)

**실측 소요**
```
LB 831 · 전량 풀링 (4090)      6시간 39분   v2 31분 + 문샷 6h08m + 제출 19초
test 2,000 · 캐스케이드        **4090 ~17시간 · RTX PRO 6000 ~9.5시간**
```
> ⚠ **캐스케이드의 시간 절감은 작다.** v2 가 확신하는 **쉬운** 문항을 생략하므로
> 문샷에 남는 문항이 전부 비싸다 — 선별분 **0.916 분/문항** vs 전량 평균 0.443 (**2.07배**).
> 문항 수는 53% 줄지만 시간은 10~15%만 준다. **4090 급은 D-Day 부적합.**

**조립 단계 실측 (2,000행)**: 선별 0.35초 · 조립 **0.37초** — 합성 병리 데이터
(거대 지수·절단·10³⁰ 정수) 주입 검증 통과. 8/16 에 90분+ 무한 정지했던 단계다.

### ⚠ 하드웨어에 따른 재현 범위

**샘플러 경로가 GPU 세대에 따라 달라집니다.** Blackwell 에서는 `VLLM_USE_FLASHINFER_SAMPLER=0`
이 필수인데, 이는 **비-Blackwell 장비와 문항별로 다른 표를 만들 수 있음**을 뜻합니다.

```text
결정성이 보장되는 범위   **같은 GPU 세대 · 같은 환경변수 · 같은 청크 크기**
보장되지 않는 범위      다른 GPU 세대 간 (샘플러 경로가 다름)
```

따라서 재현 검증 시 **제출물을 만든 장비와 동일 세대**를 사용하시기를 권합니다.
아래에 최종 제출에 실제로 쓴 환경을 기록합니다.

```text
D-Day 실행 장비   (8/31 실행 후 기록)
환경변수          (동일)
생성물 해시        (동일)
```

### 집계 규칙의 정확한 정의

`src/eval/score.py: pooled_vote()` — 순서 중립 동률 정책:

```
① 총 득표 최대            ② 동률이면 주 모델 지지표 많은 쪽
③ 그래도 동률이면 작은 값
```

등장순 정책은 파일 연결 순서에 따라 83.53 ↔ 83.32 로 흔들렸다. 현 정책은 양방향 동일하다.

---

## 5. 가중치 소재

| 모델 | 크기 | 로컬 | HF (private) |
|---|---|---|---|
| `exp-004_star_v2` (LoRA) | 468MB | `checkpoints/exp-004_star_v2/` | `ahnjun0/dlc-artifacts:exp-004-v2-adapter` |
| `exp-015_moonshot` (ep1, full) | 5.8GB | `checkpoints/exp-015_ep1/` | `ahnjun0/dlc-artifacts:exp-015-moonshot` |

전 체크포인트 **bf16** 저장. 그 외 실험 어댑터도 동일 리포에 보존되어 있다.

### 아카이브 복원 (2026-08-18 — GPU 인스턴스 반납 시 전량 백업)

> ⚠ **경로 함정 (2026-08-22 실측)**: 주 모델 어댑터는 `archive/ckpt/` 아래가 **아니다**.
> `allow_patterns=["exp-004-v2-adapter/*"]` 로 받아야 한다 —
> `archive/ckpt/exp-004_star_v2` 로 찾으면 **오류 없이 빈 결과**가 나온다.
> 리포 전체 파일 목록을 먼저 조회해 실제 경로를 확인할 것.

성능·수확·다양성 축이 모두 닫혀 대여 서버를 반납했다. **서버에 있던 모든 산출물은
HF private `ahnjun0/dlc-artifacts` 에 보존**되어 있다 (총 **46.7 GB · 664 파일**).

```
archive/data-processed/     학습 데이터 94개  (sft_v2.jsonl · moonshot_r1.jsonl 등)
archive/experiments/        판정 근거 생성물 285개 (모든 gens.jsonl · 리허설 제출물)
archive/ckpt/               폐쇄 축 어댑터 (d1·d2·d4·b03·b05·pm1dpo)
exp-004-v2-adapter/         **D-Day 주 모델** (0.49 GB)
exp-015-moonshot/           **D-Day 파트너** (6.65 GB)
```

복원 예:

```bash
huggingface-cli download ahnjun0/dlc-artifacts \
  --include "archive/experiments/exp-058_det_baseline/*" --local-dir ./restore
```

**알려진 결손 1건**: `archive/ckpt/exp-055_expblocks/` 는 업로드 중 인스턴스 종료로
**가중치 본체(`new_blocks.safetensors` 308MB)만** 올라갔고 config·tokenizer 류가 빠졌다.
exp-055 는 **폐쇄된 축**(identity block expansion — 배포 이득 미재현)이라
재현·D-Day 어디에도 쓰이지 않는다.

---

## 6. 평가 재현

```bash
# val460 채점 (maj@32 · pass@32 · 출처별 분리)
python src/inference/make_submission.py --gens <gens.jsonl> \
  --eval data/processed/val_split_corrected.csv

# 게이트 판정 (시드별 짝 비교 + 문항 부트스트랩 P + pass@32 가드)
python src/eval/compare_models.py --ctrl <기준선.jsonl> --exp <후보.jsonl>
```

**채택 기준**: 문항 부트스트랩 **P(이득) ≥ 95%**, 최소 2시드(기준선은 4시드 기대값).
**시드 간격은 100 이상** — vLLM 은 자식 시드를 `seed..seed+n-1` 로 만들어
`n=32` 에서 s42 와 s44 는 30/32 가 겹친다.

---

## 7. 재현 시 알려진 함정 (전부 실측으로 겪은 것)

1. **`parse.py` 의 지수 상한** — 모델 출력의 `1.00e23610081082016` 이 `Fraction(10)**23조` 를
   유발해 831문항 파이프라인이 **36.7% 지점에서 무한 정지**했다. `abs(exp) > 10_000 → None`
   가드로 90분+ → 18초. **모델 출력 수치를 지수·반복 연산에 그대로 넣지 말 것.**
2. **거대정수** — `set_int_max_str_digits` 없이는 파트너 사용 시 제출이 중단된다.
3. **`--resume` 필수** — 청크 단위 append+flush. 없으면 장애 한 번에 6시간 런이 소실된다.
4. **디스크** — 100GB+ 권장. 50GB 에서 두 번 사고. `snapshot_download` 후 복사하면
   같은 용량이 두 벌 남으므로 **이동**할 것.
5. **컨테이너 재시작** — 대여 인스턴스가 예고 없이 재시작된다(1시간 반에 2회 실측).
   `dday_guard.sh` 가 cron 으로 자동 재개한다.
6. **파일 완성도는 행 수로 확인** — 모델마다 생성 길이가 4~5배 달라 바이트 크기는 무의미하다.
7. **HF 어댑터 경로** — v2 는 `exp-004-v2-adapter/` 이지 `archive/ckpt/` 가 아니다.
   잘못된 경로는 **오류 없이 빈 결과**를 낸다 (5절 참조).
8. **신규 인스턴스 5단계 함정** — Vast 이미지는 `torch+cu128` 만 있고 vllm 은 CUDA13 빌드다.
   `--reinstall-package` 로 cu130 재설치 + torchvision/torchaudio **버전 동시 고정** +
   torchcodec 제거 + `transformers==5.14.1` 고정. 상세는 `scripts/run_dday_cascade.sh` 1절.
   `pytest` 도 이미지에 없다. **회선이 빠르면 전 과정 8분**(8/22 실측).
9. **완료 판정을 로그 문자열로 하지 말 것** — vLLM 의 `deep_gemm` 경고 텍스트에 `Traceback`
   이 들어 있어 감시자가 **정상 실행을 실패로 오판**했다(8/22). **산출물 행 수**로 판정한다.
10. **스크립트에 서버 주소를 박아두지 말 것** — 반납된 호스트가 남아 백업이 조용히 실패했다.
   `backup_to_hf.sh` 는 `DLC_SSH` 로 분리했다.

---

## 8. 규칙 준수 확인

| 조항 | 준수 내용 |
|---|---|
| 4.1 | 베이스는 `Qwen/Qwen2.5-3B-Instruct` 단일. 타 모델 가중치 병합 없음 |
| 4.2 | 사용 기법 = LoRA(PEFT) · Full FT · SFT · DPO — 전부 열거된 허용 기법 |
| 4.3 / 6.a | 사전학습 없음(fine-tuning 만). 추론 시 인터넷·외부 모델·**코드 실행 없음** |
| 5.1b / 5.3b | test 문항 학습 미사용. 상용 API 로 test 답 생성 안 함. LB probing 없음 |
| 5.2 | 외부 데이터는 전부 **무료 공개** (DATA_SOURCES.md). 유료·비공개 데이터 없음 |
| 6.c | 집계는 다수결(Self-Consistency) — 명시적 허용 |
| 8.2 | 본 문서 + 링크 문서로 코드·가중치·데이터·환경·방법론 제출 |

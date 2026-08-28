# 제5회 대학 연합 딥러닝 챌린지 2026 — 제출 저장소

**참가자** 안준영 (개인전) · **베이스 모델** `Qwen/Qwen2.5-3B-Instruct` (고정)
**과제** 수학 문제 → 정수 정답 (Exact Match) · **리더보드** 0.80385 (668/831)

최종 제출 구성은 **3단 적응형 캐스케이드**입니다. 세 모델 모두 지정 베이스에서 파생됩니다.

```
test 문항
  └─ ① STAR 32표          최다 득표 ≥22  →  확정
        └─ ② + LONGCOT 32표 (64표)   1·2위 격차 >5  →  확정
              └─ ③ + STAR-OLY 32표 (96표)  →  확정
```

**분기 변수는 「표 구조」뿐입니다** — 문제의 주제·출처·정답을 보지 않습니다.

| 역할 | 실험 ID | 형태 | 학습 데이터 |
|---|---|---|---|
| ① 주 모델 `STAR` | `exp-004_star_v2` | LoRA | 대회 train 자기증류 (STaR) |
| ② 파트너 `LONGCOT` | `exp-015_moonshot` | full FT | R1 long-CoT 증류 |
| ③ 증원 `STAR-OLY` | `exp-035_contest` | LoRA | 외부 경시 풀 자기증류 |

---

## 규칙 준수

### 베이스 모델은 `Qwen/Qwen2.5-3B-Instruct` 하나뿐입니다

세 모델 전부 이 베이스에서 파생된 어댑터·미세조정이며, **가중치 병합(merge)은 없습니다.**

```bash
git grep -hoE "Qwen/Qwen[0-9.]+-[A-Za-z0-9.-]+" -- '*.py' '*.sh' | sort | uniq -c
#   → Qwen/Qwen2.5-3B-Instruct 만 나옵니다
```

### 추론 시 외부 호출이 없습니다

```bash
git grep -nE "requests\.|httpx|openai|urllib|api_key|https?://" -- src/ | grep -v "^docs/"
#   → 검출 0건
```

추론은 로컬 vLLM 만 사용합니다 (`src/inference/generate.py`).
집계·제출물 생성은 순수 파이썬입니다 (`src/eval/{parse,score,build_cascade}.py`).

### 사용한 외부 데이터

[`data/external/DATA_SOURCES.md`](data/external/DATA_SOURCES.md) 에 출처·라이선스·용도를
기록했습니다. 전부 **무료로 동등하게 접근 가능한 공개 데이터**입니다.

---

## 환경

```bash
# GPU (학습·추론) — CUDA >= 13.0 · VRAM 48GB 권장
pip install -r requirements-gpu.txt

# 로컬 (평가·분석만)
pip install -r requirements-local.txt
pytest tests/ -q
```

핵심 버전: `torch 2.11+cu130` · `vllm 0.26.0` · `transformers 5.14.1` · `trl 1.9.2` · `peft 0.20.0`

> ⚠ 호스트 CUDA 드라이버가 13.0 미만이면 `torch.cuda.is_available()` 이 True 여도
> 실제 커널에서 실패합니다. Blackwell(SM 12.x) 에서는 FlashInfer 샘플러를 꺼야 하며,
> `scripts/lib/vllm_env.sh` 가 자동으로 감지합니다.

## 실행

**전체 절차는 [`REPRODUCE.md`](REPRODUCE.md) 에 실행 가능한 커맨드로 정리했습니다** —
환경 · 데이터 생성 · 학습(3모델) · 추론 · 제출물 생성 · 알려진 함정까지.

### 최단 경로 — 학습된 가중치로 제출물 재현

```bash
bash scripts/run_dday_cascade.sh          # 가중치는 스크립트가 HF 에서 자동 회수
```

`DDAY_IN` 으로 입력 CSV, `DDAY_OUT` 으로 출력 경로를 지정합니다.
2,000문항 실측 소요는 약 12시간(RTX PRO 5000 48GB 기준)입니다.

### 처음부터 학습

`REPRODUCE.md` §2(데이터) → §3(학습) → §4(추론) 순서를 따르시면 됩니다.

## 방법론

[`docs/methodology.md`](docs/methodology.md) — 판정 규칙(사전 등록 · 문항 부트스트랩
P≥95% · 다중 시드 · 문항 폴드 교차적합 · LB 짝 검정), 실험 설계, 자기 평가.

핵심 관측 하나를 남깁니다 — `maj@k` 에서 점수를 지배하는 것은 능력(`pass@32`)이 아니라
**표의 분산**입니다. 능력이 비슷한 7개 모델에서 `pass@32 ↔ maj@32` 는 **r = +0.134**,
`n_eff ↔ maj@32` 는 **r = −0.920** 이었습니다. 캐스케이드의 세 자리가 각각 다른 것을
요구하는 이유가 여기서 나옵니다.

## 저장소 구성

```
src/inference/    생성(vLLM) · 제출 CSV
src/eval/         정수 파싱 · 다수결 · 캐스케이드 조립/선별 · 엄격 검증기
src/train/        LoRA · full fine-tuning
src/data/         홀드아웃 분할 · STaR 데이터 빌드 · 경시 풀 구축
scripts/          최종일 실행 · 자동 재개 감시자 · 환경 구축
tests/            파서·집계·환경 회귀 테스트
```

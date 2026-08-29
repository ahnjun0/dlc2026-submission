내	

# 사용 데이터 목록 (최종 제출 시 명시 의무)

## 검증용 (학습에 사용하지 않음)

| 이름    | 출처 (HF)                          | 라이선스                     | 규모(정수답 필터 후) | 용도                    |
| ------- | ---------------------------------- | ---------------------------- | -------------------- | ----------------------- |
| aime    | AI-MO/aimo-validation-aime         | Apache-2.0                   | 90/90                | 외부 검증 (최고난도)    |
| amc     | AI-MO/aimo-validation-amc          | Apache-2.0                   | 83/83                | 외부 검증               |
| math_l4 | AI-MO/aimo-validation-math-level-4 | Apache-2.0 (MATH 원본은 MIT) | 744/754              | 외부 검증 (중상 난이도) |
| math_l5 | AI-MO/aimo-validation-math-level-5 | Apache-2.0 (MATH 원본은 MIT) | 719/721              | 외부 검증 (상 난이도)   |

- 구축 스크립트: `src/data/build_external_valsets.py` (정답이 정수인 문항만 유지)
- 파일: `data/external/valsets/*.csv` (git 미추적 — 스크립트로 재생성 가능)

## 분석용 (출처 식별 — 2026-07-31)

train 17,000문항의 출처 매칭 결과 (`src/data/match_train_sources.py` → `data/processed/train_source_map.csv`):

- **총 87.7% (14,909문항) 출처 확인.** 커버율: NuminaMath-1.5 84.7% > Orca-Math 47.1% > MetaMathQA 29.4% > GSM8K 28.3% (상호 중복 큼 — NuminaMath-1.5가 orca_math/metamath 서브셋을 포함)
- NuminaMath-1.5 내부 출처별: orca_math 11,547 / synthetic_math 2,972 / olympiads 2,003 / cn_k12 1,599 / metamath 992 / aops_forum 341 / cn_contest 189 / amc_aime 71 등
- ~~미매칭 12.3%~~ → **2026-08-01 웹 조사(Sonnet 에이전트) + 데이터셋 검증으로 대부분 해소**:
  - **Omni-MATH** (KbsdJames/Omni-MATH): 1,188건 매칭 — NuminaMath 미수록 경시대회 문제군
  - **GSM-Plus** (qintongli/GSM-Plus): 414건 매칭 — GSM8K 수치 변형(perturbation) 증강 문제. "숫자만 바뀐 GSM8K"라 원본 매칭에 실패했던 것
  - 최종 미매칭 **656건(3.9%)**: ARML 릴레이("Let $T=...") + HMMT류 추정 (공개 데이터셋 미확인)
  - **총 출처 확인율 96.1%** (16,344/17,000)
- **결론: 대회 train ≈ NuminaMath-1.5 분포의 표본.** SFT 주력 데이터로 NuminaMath-1.5(정수답 필터)가 도메인 최적.

## 분석·출처검증용 데이터셋 명세 (2026-08-01 추가)

| 이름      | 출처 (HF)           | 라이선스        | 규모   | train 매칭 | 용도                                                             |
| --------- | ------------------- | --------------- | ------ | ---------- | ---------------------------------------------------------------- |
| GSM-Plus  | qintongli/GSM-Plus  | CC-BY-SA-4.0    | 10,552 | 414건      | 출처 검증. GSM8K 수치변형 — 변형 출제 대비 참고                 |
| Omni-MATH | KbsdJames/Omni-MATH | MIT (경시 원문) | 4,428  | 1,188건    | 출처 검증 +**olympiad 슬라이스 SFT 보충 후보** (해설 포함) |
| SVAMP     | ChilleD/SVAMP       | MIT             | 700    | 3건        | 배제 (매칭 미미)                                                 |
| ASDiv     | EleutherAI/asdiv    | CC-BY-NC-4.0    | 2,305  | 12건       | 배제 (매칭 미미 +**NC 라이선스 — 학습 사용 금지**)        |

## 상용 API (학습 데이터 구축용 — 규칙 5.3.a 명시 허용, 최종 제출 시 기재)

| API                     | 모델             | 용도                                               | 파일럿 정답률 (최고난도 20문항) |
| ----------------------- | ---------------- | -------------------------------------------------- | ------------------------------- |
| Elice ML API            | GPT-5.4          | v4 증류 1단 (hard 4,601 해설 생성) + 라벨 3자 대조 | 12/20                           |
| Elice ML API            | Claude Opus 4.5  | v4 증류 2단 (1단 불일치분 에스컬레이션)            | 12/20                           |
| Anthropic (Claude Code) | Claude 최신 세대 | 증류 3단 + 라벨 블라인드 검증 + 오답 분류          | 17/20                           |
| Upstage                 | Solar Pro 2      | 파일럿만 (최하위로 탈락)                           | 9/20                            |

## 데이터 위생 산출물 (data/processed/)

- `train_label_suspects.csv` (748건): 모델 강수렴(≥6/8) ≠ gold. teacher 대조 누적 **1,182건 교정 확정** (블라인드 정밀도 94%)
- `train_broken_ids.csv` (145건): 구조적 손상(외부 이미지 링크 127 / 지시문 잔재 15 / 목차 잔재 2 / 수동 1) → 모든 학습 데이터에서 하드 제외
- `train_numina_src.csv`: 문항별 NuminaMath 내부 출처 (출처별 평가 리포트용)
- `star_solve_stats.csv`: 문항별 8샘플 해결률 (난이도 지도)
- `val_topics.csv` (8/2): val 500 주제 분류 — 조합론 36.4%가 최약 주제 (표적 보강 축)

## 학습용 데이터 계보

- **sft_v1** (exp-003a, 폐기): NuminaMath 해설 직접 143k — 짧은 해설 스타일이 성능 회귀 유발
- **sft_STAR** (exp-004, LB 0.735): STaR 자기생성 18,076 + 외부 보충 1,109 = 19,185건
- **sft_v3** (exp-005, 이득 없음으로 폐기): 라운드2 자기생성 15,073 + 보충 1,217 = 16,290건 — maj@32 74.6 (반복 축 폐쇄 근거)
- **sft_v4** (exp-006, 게이트 미달로 폐기): 정화 STaR 15,811 + teacher 1,730, 위생 3층 — maj@32 73.4% (-1.6). teacher 혼합 자체가 용의자
- **sft_v5a/b/c** (예정, 8/3 공식 교정 후): 3-arm — 순수 교정 STaR / +teacher / +rationalization
- **genselect_train** (exp-008, 축 폐쇄로 보관): 판정 9,324쌍 — 선택기 실험용 (재사용 없음)
- **external_star_pool.csv** (exp-024에 사용): 외부 문항 25,000 (대회 train·LB dedup 완료). **출처 사후 특정 완료 (2026-08-10, `src/data/identify_ext_pool.py` → `data/processed/ext_pool_sources.csv`)** — 빌드 스크립트가 남지 않아 출처 불명이던 것을 해소:
  - **AI-MO/NuminaMath-1.5 100.0% (24,993/25,000)**, 그중 66건은 Omni-MATH와도 중복, 미매칭 7건
  - NuminaMath 내부 출처: orca_math 12,274 / synthetic_math 4,725 / olympiads 3,838 / cn_k12 2,541 / metamath 2,187 / aops_forum 1,462 / cn_contest 1,026 / amc_aime 785
  - **함의**: 대회 train 자체가 NuminaMath-1.5의 표본이므로, exp-024는 "새 분포"가 아니라 **"같은 분포에서 더 많이"**였다

## 공식 검수 산출물 (2026-08-03, 주최측 배포)

- `data/raw/deep_chal_math_leaderboard_filtered.csv`: 새 LB 831문항 (answer 컬럼 제거) — 이후 모든 LB 추론의 기준
- `data/raw/train_filtered_ids.csv`: 공식 train 오류 627건 (id,answer,question 전체 행) — 모든 학습 데이터에서 하드 제외
- `data/processed/official_errata.csv`: 위를 rescore_all 형식(exclude)으로 변환
- `data/processed/val_split_corrected.csv`: 공식 27건 제외 val 473문항 — **새 표준 평가셋**
- **sft_v5a** (8/3): STAR 레시피 + 공식 검수 위생 — STaR 17,632 (교정 복권 1,506) + 보충 724 = 18,356건. 제외 901 (공식 627 ∪ 손상 ∪ 미교정 의심 292), 교정 라벨 1,182 적용, teacher 해설 미혼합

## LONGCOT 학습 데이터 (2026-08-04 구축)

| 이름             | 출처 (HF)                | 라이선스   | 규모                              | 용도                           |
| ---------------- | ------------------------ | ---------- | --------------------------------- | ------------------------------ |
| OpenR1-Math-220k | open-r1/OpenR1-Math-220k | Apache-2.0 | 93,733 → 필터 후**25,953** | LONGCOT long-CoT full FT (조건부) |

- 필터: 정수답 50,905 제외 → math_verify 검증 + 추론 완결 생성물만 → 24k자 초과 7,375 제외 → **LB 831 정규화 중복 47건 기계적 제거** (규칙 6 dedup 실증)
- 빌드: `src/data/build_moonshot_data.py` → `data/processed/moonshot_r1.jsonl` (서버, 재현 가능)

## API 자원 변경 (2026-08-06)

- **Elice ML API 사용 불가** (GPT-5.4 / Claude Opus 4.5). 기 생성 산출물은 보존: `teacher_gpt54.jsonl`(4,601), `teacher_opus45.jsonl`(2,189), 라벨 교정 1,182건 — 최종 제출 시 데이터 출처 목록에는 그대로 기재 필요(사용 사실은 변하지 않음)
- 현행 대안: Claude Code 서브에이전트(소량 고판단) / 로컬 GPU + R1-Distill-7B(대량, 판정 품질 측정 예정) / Upstage Solar Pro 2(보조) / Alibaba(미검증)

## SmallDoge/SmallThoughts (2026-08-26 · exp-120)

| 항목 | 내용 |
| --- | --- |
| 출처 | https://huggingface.co/datasets/SmallDoge/SmallThoughts |
| 라이선스 | **Apache-2.0** |
| 접근성 | HF 공개 · 무료 · 인증 불요 → **규칙 5.2 «모든 참가자가 무료로 동등하게 접근» 충족** |
| 용도 | **중간 길이(concise) CoT** 생성 체제를 만들기 위한 SFT 데이터 |
| 관련 연구 | Difficulty-Aware Distillation (LiteCoT) — 난이도 판단 후 핵심 단계만 남긴 압축 CoT |

**정제 (`src/data/build_concise_cot.py`)**
```text
총 100,000 → **채택 19,009**
  비정수 제외      68,571   `\boxed{}` 안이 **순수 정수**인 것만 (중괄호 균형 파싱)
  코드/TIR 제외        24   추론 시 코드 실행 금지(4.3) — 코드를 쓰고 결과를 지어내는 모델 방지
  길이 구간 제외    12,333   1,800~6,000자만 (STAR 1,412 · LONGCOT 10,425 사이의 빈 체제)
  중복 제거            63   대회 train ∪ val ∪ leaderboard 와 정규화(md5) 대조
시스템 프롬프트는 **우리 것(184자)으로 교체** — 학습·추론 불일치는 `exp-010` 을 죽인 원인이다
결과: assistant 길이 중앙 **2,406자** · 구조 마커 `<|begin_of_thought|>` **100%**
```

> **주의 (제출 문서에 함께 적을 것)**: 원 논문은 일부 평가에서 **5회 중 최고 점수**를
> 보고하므로 효과가 낙관적으로 부풀 수 있다. 이 데이터는 **«그런 체제가 존재한다»의 근거**이지
> 우리 데이터에서 개선을 예측하는 근거가 아니다.

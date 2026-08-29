# 대회 규정 원문 (2026-08-29 직접 확보)

출처 ① Kaggle 규칙 페이지 `competitions/deep-learning-challenge-2026/rules` (로그인 후 직접 열람)
출처 ② Discord `#공지사항` 채널 (공지 4건 전문)

**이 문서는 요약이 아니라 조항 대조용 원문 발췌다.** `src/eval/rules_audit.py` 가 이 조항 번호를 참조한다.

## 4. 모델 규칙

```text
4.1a  모든 참가자는 Qwen/Qwen2.5-3B-Instruct 를 유일한 출발점으로 사용해야 합니다.
4.1b  다른 모델(Qwen2.5-Math, DeepSeek-R1, Llama, GPT 등)을 베이스 모델로 사용하거나,
      다른 모델의 가중치를 병합(merge)하는 것은 금지됩니다.
4.2   허용: Full Fine-Tuning, LoRA, QLoRA 등 PEFT / SFT / RL(GRPO, DPO, PPO, KTO 등)
             / 데이터 증강, 커리큘럼 학습 / 양자화
4.3   금지: ① 베이스 모델 외 다른 모델의 가중치를 로드하거나 병합하는 행위
             ② **추론 시 외부 모델을 호출하여 앙상블하는 행위**
             ③ 사전 학습(Pre-training)을 처음부터 수행하는 행위 (Fine-tuning만 허용)
```

## 5. 데이터 규칙

```text
5.1a  주최 측 제공 학습 데이터를 기본으로 활용
5.1b  **test.parquet 의 문제를 학습 데이터로 사용하는 것은 금지**
5.2a  공개 데이터셋 추가 사용 자유. 단 **모든 참가자가 무료로 동등하게 접근 가능**해야 함
5.2b  유료 구독·특수 라이선스·비공개 협약 데이터 금지
5.2c  **사용한 외부 데이터셋은 최종 제출 시 목록 명시**
5.3a  **학습 데이터 구축 목적의 상용 API 사용은 허용** (예: GPT-4 로 풀이 생성, 데이터 증강)
5.3b  상용 API 로 test.parquet 문제의 답을 직접 생성하는 것은 금지
5.3c  테스트 문제를 검색 엔진이나 외부 서비스에 입력하여 답을 찾는 행위 금지
```

## 6. 추론 규칙

```text
6.a  추론 시 인터넷 접속이 차단됩니다 (외부 API 호출, 웹 검색 등 불가)
6.b  모든 추론은 제공된 환경 내에서 로컬로 수행
6.c  **Majority Voting, Self-Consistency, Best-of-N 등 테스트 타임 기법은 자유**
```

## 7·8. 평가와 제출

```text
7.2a-c  정답은 정수. 정확히 일치하면 정답. 지표 Accuracy (Exact Match)
7.2d    모델 출력에서 최종 답을 추출하는 후처리는 참가자가 직접 수행
8.1     submission.csv · ID 와 answer 두 컬럼 · 정수만 · 빈 값은 오답
8.2a    수상 후보자 제출물: 학습 코드 및 추론 코드 / 학습된 모델 가중치 /
        사용한 데이터셋 목록 및 접근 방법 / 재현 환경 설명 / 방법론 설명 문서
8.2b    **재현이 불가능한 경우 수상이 취소될 수 있습니다**
```

## 10. 부정행위

```text
10.1a  테스트 데이터의 정답을 외부에서 확보하여 제출하는 행위
10.1b  다른 참가자의 코드·모델·제출물을 무단 사용
10.1c  다중 계정으로 제출 횟수 우회
10.1d  **리더보드 점수를 이용한 정답 역추적(probing)**
10.1e  **본 규칙을 우회하기 위한 의도적인 행위**
```

## Kaggle Foundational Rules (충돌 시 우선)

```text
4.b  Submissions may not use or incorporate information from **hand labeling or human
     prediction of the validation dataset or test data records**
6.a  Private Code Sharing — Competition Period 중 코드 비공개 공유 금지
6.b  Public Code Sharing — 공개 공유를 택하면 **Kaggle 포럼/노트북에** 공유할 것
6.c  Use of Open Source — **OSI 승인 라이선스**만 사용
```

## Discord 공지 (조항과 같은 효력, 11.1 규칙 변경 조항)

```text
7/31  test.csv 유출본 폐기. 최종 평가는 8/31 별도 test dataset + 구글 폼 제출
8/01  라벨 오류 확인. LB 재평가 예고
8/03  LB 1,000 → **831문항**(169 제외), deep_chal_math_leaderboard_filtered.csv 배포
      train 오류 **627개 id** 공개(train_filtered_ids.csv) — 학습 시 제외 권고
8/28  **8/30 23:59 개발 마감** · 8/31 test 공개 → CSV + **GitHub URL** 구글 폼 제출
      **GitHub 저장소는 Public + 실행 환경·방법이 정리된 README 필수**
      "지정된 Qwen/Qwen2.5-3B-Instruct 외 타 베이스 모델 사용이나 **추론 시 외부 API 호출** 등
       규칙 위반 사항이 확인될 경우 **리더보드 순위에서 제외**"
      제출물 검증·채점 8/31~9/20 → 상위 12팀 발표(50%) → 최종 9팀 수상
```

## ⚠ 규정 원문과 실제가 다른 지점 (실측으로 확인)

```text
7.1     "Public 30% / Private 70%" — **갱신 안 된 구 내용.** 실제는 8/31 별도 test dataset
8.1b    "ID 와 answer" + 예시 prob_0001 — **실제 Kaggle 채점기는 소문자 `id` 를 요구**
        (2026-08-23 실측: `ID,answer` 로 제출하면 채점 없이 ERROR)
        최종 제출은 구글 폼 + 운영진 수기 채점이므로 당일 안내를 따를 것
```

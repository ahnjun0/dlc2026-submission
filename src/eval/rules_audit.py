"""대회 규정 자동 검증기.

`docs/rules-verbatim-2026-08-29.md` 에 기록한 조항 원문을 기계적 검사로 옮긴 것이다.
Kaggle 규칙 페이지(4~10절) · Kaggle Foundational Rules · Discord #공지사항 공지 4건이 근거다.

    python src/eval/rules_audit.py --repo <제출저장소> [--data-root <데이터가 있는 저장소>]

각 검사는 조항 번호를 달고 PASS / FAIL / WARN / MANUAL 중 하나를 낸다.
- FAIL   : 규정 위반. 고치기 전에는 제출 불가.
- WARN   : 위반은 아니나 채점관이 갸웃할 수 있는 지점. 문서로 해명돼 있어야 한다.
- MANUAL : 기계가 판정할 수 없어 사람이 확인해야 하는 항목. 근거 위치를 같이 출력한다.

**설계 원칙**: 검사는 우리에게 유리하게 기울지 않도록 «금지된 것을 찾는» 방향으로 쓴다.
증거를 찾지 못한 것을 PASS 로 부르지 않고, 무엇을 훑었는지 함께 출력한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 배포 경로 = 규칙 4.1/4.3/6.a 가 적용되는 코드. 데이터 구축 경로(5.3a)는 별도 취급 ──
DEPLOY_GLOBS = ["src/train/**/*.py", "src/inference/**/*.py", "src/eval/**/*.py",
                "scripts/run_dday_cascade.sh", "scripts/dday_guard.sh",
                "scripts/run_contest_pipeline.sh"]
DATABUILD_GLOBS = ["src/data/**/*.py"]

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# HF 모델 id 처럼 보이는 문자열. 데이터셋 id 와 구분하기 위해 별도로 걸러낸다.
HF_ID_RE = re.compile(r"[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.+-]+)[\"']")

STATUS_ORDER = {"FAIL": 0, "WARN": 1, "MANUAL": 2, "PASS": 3, "SKIP": 4}


@dataclass
class Result:
    clause: str
    name: str
    status: str
    detail: str
    evidence: list[str] = field(default_factory=list)


def files(root: Path, globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for g in globs:
        out.extend(sorted(p for p in root.glob(g) if p.is_file()))
    return out


def norm_q(s: str) -> str:
    """오염 검사용 문항 정규화 — 공백·대소문자·구두점 제거."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ══════════════════════════════════════════════════════════════════════
# 4. 모델 규칙
# ══════════════════════════════════════════════════════════════════════

def c_4_1a_base_model(repo: Path) -> Result:
    """[4.1a] 배포 경로가 Qwen2.5-3B-Instruct 외의 모델을 로드하지 않는가."""
    fs = files(repo, DEPLOY_GLOBS)
    foreign: list[str] = []
    for p in fs:
        txt = p.read_text(errors="ignore")
        for m in HF_ID_RE.finditer(txt):
            ident = m.group(1)
            if ident == BASE_MODEL or not ident.startswith(("Qwen/", "deepseek-ai/",
                                                            "meta-llama/", "mistralai/",
                                                            "microsoft/", "google/", "openai/")):
                continue
            line = txt[:m.start()].count("\n") + 1
            foreign.append(f"{p.relative_to(repo)}:{line}  {ident}")
    if foreign:
        return Result("4.1a", "배포 경로 베이스 모델", "FAIL",
                      f"베이스 외 모델 참조 {len(foreign)}건", foreign)
    return Result("4.1a", "배포 경로 베이스 모델", "PASS",
                  f"{len(fs)}개 파일에서 베이스 외 모델 로드 없음 "
                  f"(base={BASE_MODEL})")


def c_4_1b_merge(repo: Path) -> Result:
    """[4.1b/4.3①] 다른 모델 가중치를 병합하는 코드가 없는가."""
    pat = re.compile(r"merge_and_unload|merge_adapter|add_weighted_adapter|"
                     r"model_?soup|task_arithmetic|slerp|ties_merge|dare_")
    hits = []
    for p in files(repo, DEPLOY_GLOBS + DATABUILD_GLOBS):
        for i, ln in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if pat.search(ln):
                hits.append(f"{p.relative_to(repo)}:{i}  {ln.strip()[:90]}")
    if hits:
        return Result("4.1b", "가중치 병합", "MANUAL",
                      "병합 관련 호출 발견 — 베이스+자기 LoRA 병합은 합법, 타 모델 병합은 위반", hits)
    return Result("4.1b", "가중치 병합", "PASS", "병합 API 호출 자체가 없음")


def c_4_3_2_inference_network(repo: Path) -> Result:
    """[4.3②/6.a/6.b] 추론 경로에 네트워크 호출·외부 모델 앙상블이 없는가."""
    pat = re.compile(r"\b(requests\.|httpx\.|urllib|aiohttp|openai|anthropic|"
                     r"OpenAI\(|api_key|API_KEY|\.chat\.completions|curl |wget )")
    allow = re.compile(r"HF_TOKEN|huggingface|hf_hub|snapshot_download|"
                       r"^\s*#|backup|upload")  # 가중치 회수·백업은 추론 전이다
    hits = []
    for p in files(repo, ["src/inference/**/*.py", "src/eval/**/*.py"]):
        for i, ln in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if pat.search(ln) and not allow.search(ln):
                hits.append(f"{p.relative_to(repo)}:{i}  {ln.strip()[:90]}")
    if hits:
        return Result("4.3②", "추론 시 외부 호출", "FAIL",
                      f"추론 경로 네트워크 호출 {len(hits)}건", hits)
    return Result("4.3②", "추론 시 외부 호출", "PASS",
                  "src/inference·src/eval 전체에 HTTP/API 호출 없음 "
                  "(vLLM 로컬 추론만)")


def c_4_3_3_pretrain(repo: Path) -> Result:
    """[4.3③] 처음부터 사전학습하는 코드가 없는가 (from_config / 랜덤 초기화)."""
    bad = re.compile(r"from_config\(|AutoModelForCausalLM\(\s*config|init_weights\(\s*\)")
    good = re.compile(r"from_pretrained")
    hits, oks = [], []
    for p in files(repo, ["src/train/**/*.py"]):
        txt = p.read_text(errors="ignore")
        if bad.search(txt):
            hits.append(str(p.relative_to(repo)))
        if good.search(txt):
            oks.append(str(p.relative_to(repo)))
    if hits:
        return Result("4.3③", "from-scratch 사전학습", "FAIL", "랜덤 초기화 흔적", hits)
    return Result("4.3③", "from-scratch 사전학습", "PASS",
                  f"학습 스크립트 {len(oks)}개 전부 from_pretrained 로 시작", oks)


def c_4_2_techniques(repo: Path) -> Result:
    """[4.2] 사용한 학습 기법이 열거된 허용 목록 안인가."""
    allowed = {"Full Fine-Tuning": r"sft_full|full[ _-]?ft",
               "LoRA/PEFT": r"LoraConfig|get_peft_model|peft",
               "SFT": r"SFTTrainer|SFTConfig",
               "RL(GRPO/DPO/PPO/KTO)": r"GRPOTrainer|DPOTrainer|PPOTrainer|KTOTrainer",
               "데이터 증강·커리큘럼": r"augment|curriculum|oversample",
               "양자화": r"bitsandbytes|load_in_4bit|quantiz"}
    trainer = re.compile(r"([A-Z][A-Za-z]*Trainer)\b")
    found, unknown = [], set()
    txts = {p: p.read_text(errors="ignore") for p in files(repo, ["src/train/**/*.py"])}
    blob = "\n".join(txts.values())
    for name, pat in allowed.items():
        if re.search(pat, blob, re.I):
            found.append(name)
    known_tr = {"SFTTrainer", "GRPOTrainer", "DPOTrainer", "PPOTrainer", "KTOTrainer", "Trainer"}
    for m in trainer.finditer(blob):
        if m.group(1) not in known_tr:
            unknown.add(m.group(1))
    if unknown:
        return Result("4.2", "허용 기법", "MANUAL",
                      "목록에 없는 Trainer 사용 — 4.2 열거식 목록과 대조 필요", sorted(unknown))
    return Result("4.2", "허용 기법", "PASS",
                  "사용 기법이 전부 4.2 열거 목록 안: " + ", ".join(found))


# ══════════════════════════════════════════════════════════════════════
# 5. 데이터 규칙
# ══════════════════════════════════════════════════════════════════════

def c_5_1b_contamination(repo: Path, data: Path, train_sets: list[Path]) -> Result:
    """[5.1b/10.1a] 평가 문항이 학습 데이터에 들어가 있지 않은가.

    최종 test 는 8/31 에야 공개되므로 검사 대상은 **LB 원본 1,000문항 전량**이다.
    (831 필터본만 대조하면 제외된 169문항을 놓친다 — 8/28 에 실제로 저질렀던 실수)
    val_split 은 규칙이 아니라 우리 자체 규율이므로 별도 검사한다.
    """
    lb_files = [data / "data/raw/deep_chal_math_leaderboard.csv",
                data / "data/raw/deep_chal_math_leaderboard_filtered.csv"]
    lb: set[str] = set()
    used = []
    for f in lb_files:
        if not f.exists():
            continue
        used.append(f.name)
        with f.open(newline="") as fh:
            for row in csv.DictReader(fh):
                q = next((v for k, v in row.items() if k.strip().lower() == "question"), None)
                if q:
                    lb.add(norm_q(q))
    if not lb:
        return Result("5.1b", "평가 문항 오염", "SKIP", "LB 원본을 찾지 못함")

    avail = [p for p in train_sets if p.exists()]
    missing = [p.name for p in train_sets if not p.exists()]
    if not avail:
        return Result("5.1b", "평가 문항 오염", "SKIP",
                      "학습 데이터 파일 없음 (HF 백업에서 회수 필요): " + ", ".join(missing))

    hits, n, hit_keys = [], 0, []
    for p in avail:
        with p.open() as fh:
            for i, line in enumerate(fh, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                msgs = rec.get("messages") or []
                q = next((m.get("content", "") for m in msgs if m.get("role") == "user"),
                         rec.get("problem") or rec.get("question") or "")
                k = norm_q(q)
                if k in lb:
                    hits.append(f"{p.name}:{i}")
                    hit_keys.append(k)
    # 채점 대상(필터본 831)에 든 것만 실질 위험이다. 제외된 169문항은 채점되지 않는다.
    scored = set()
    ff = data / "data/raw/deep_chal_math_leaderboard_filtered.csv"
    if ff.exists():
        with ff.open(newline="") as fh:
            scored = {norm_q(r["question"]) for r in csv.DictReader(fh) if r.get("question")}
    n_scored = sum(1 for k in hit_keys if k in scored)
    ev = [f"대조 기준: {', '.join(used)} → 고유 문항 {len(lb)}개",
          f"검사한 학습 샘플: {n:,}개 ({', '.join(p.name for p in avail)})",
          f"일치 {len(hits)}건 — **그중 채점 대상(831 필터본) {n_scored}건**"]
    if missing:
        ev.append(f"※ 미검사: {', '.join(missing)}")
    # 규칙 5.1b 가 금지하는 것은 **test.parquet** 이다. LB 학습 사용 금지 조항은 없다.
    # 다만 10.1e(취지 우회) 와 우리 자체 dedup 규율이 걸리므로 수치와 함께 보고한다.
    if n_scored:
        return Result("5.1b", "평가 문항 오염", "WARN",
                      f"채점 대상 LB 문항 {n_scored}건이 학습 데이터에 존재. "
                      "**조항 위반은 아니다** — 5.1b 가 금지하는 것은 test.parquet 이고 "
                      "LB 문항의 학습 사용을 금지한 조항은 없다. 다만 자체 dedup 규율 "
                      "미준수이며 영향 크기를 문서에 명시해야 한다", ev + hits[:8])
    st = "PASS" if not missing else "WARN"
    return Result("5.1b", "평가 문항 오염", st,
                  f"채점 대상 831문항과의 일치 0건 / {n:,} 샘플", ev)


def c_val_leak(repo: Path, data: Path, train_sets: list[Path]) -> Result:
    """[자체 규율] val_split 문항이 학습 데이터에 없는가 — 규칙이 아니라 판정 무결성."""
    vf = data / "data/processed/val_split_corrected.csv"
    if not vf.exists():
        vf = data / "data/processed/val_split.csv"
    if not vf.exists():
        return Result("자체", "val 누설", "SKIP", "val_split 없음")
    with vf.open(newline="") as fh:
        val = {norm_q(r["question"]) for r in csv.DictReader(fh) if r.get("question")}
    avail = [p for p in train_sets if p.exists()]
    if not avail:
        return Result("자체", "val 누설", "SKIP", "학습 데이터 없음")
    hits, n = [], 0
    for p in avail:
        with p.open() as fh:
            for i, line in enumerate(fh, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                msgs = rec.get("messages") or []
                q = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                if norm_q(q) in val:
                    hits.append(f"{p.name}:{i}")
    if hits:
        return Result("자체", "val 누설", "WARN",
                      f"{len(hits)}샘플 — 규칙 위반이 아니다(val 은 train 에서 우리가 만든 "
                      "holdout 이고 train 사용은 5.1a 로 허용된다). 판정 무결성 문제이므로 "
                      "영향 크기를 실측해 문서에 남길 것", hits[:10])
    return Result("자체", "val 누설", "PASS",
                  f"val {len(val)}문항 대 학습 {n:,}샘플 일치 0건")


def c_5_2c_sources(repo: Path) -> Result:
    """[5.2c/5.2a/5.2b/Foundational 6.c] 외부 데이터 목록·접근성·라이선스."""
    ds = repo / "data/external/DATA_SOURCES.md"
    if not ds.exists():
        return Result("5.2c", "외부 데이터 목록", "FAIL", "DATA_SOURCES.md 없음")
    txt = ds.read_text()
    # 코드가 실제로 참조하는 HF 데이터셋 id 를 추출해 문서와 대조
    dpat = re.compile(r"load_dataset\(\s*[\"']([^\"']+)[\"']")
    used = set()
    for p in files(repo, DATABUILD_GLOBS + DEPLOY_GLOBS):
        used.update(dpat.findall(p.read_text(errors="ignore")))
    undocumented = sorted(d for d in used if d.split("/")[-1] not in txt and d not in txt)
    open_lic = re.compile(r"MIT|Apache|CC[- ]?BY|CC0|BSD|ODC|Open Data", re.I)
    paid = re.compile(r"유료|구독|비공개|proprietary|paid", re.I)
    ev = [f"코드가 참조하는 데이터셋 {len(used)}개: " + ", ".join(sorted(used))]
    if undocumented:
        return Result("5.2c", "외부 데이터 목록", "FAIL",
                      "문서에 없는 데이터셋", ev + undocumented)
    if paid.search(txt) and "금지" not in txt:
        return Result("5.2b", "외부 데이터 접근성", "MANUAL",
                      "유료/비공개 언급 — 5.2b 위반 여부 확인", ev)
    if not open_lic.search(txt):
        return Result("5.2c", "외부 데이터 목록", "WARN",
                      "라이선스 표기가 확인되지 않음 (Foundational 6.c 는 OSI 승인만 허용)", ev)
    return Result("5.2c", "외부 데이터 목록", "PASS",
                  "코드가 쓰는 데이터셋 전부 DATA_SOURCES.md 에 기재 + 개방 라이선스 표기", ev)


def c_5_3bc_api(repo: Path) -> Result:
    """[5.3b/5.3c] 상용 API 스크립트가 평가 문항을 읽지 않는가."""
    apipat = re.compile(r"openai|anthropic|api\.upstage|chat\.completions|dashscope", re.I)
    evalpat = re.compile(r"leaderboard|test\.parquet|test\.csv|deep_chal_math_test")
    hits, api_files = [], []
    for p in files(repo, ["src/**/*.py", "scripts/**/*.sh"]):
        txt = p.read_text(errors="ignore")
        if not apipat.search(txt):
            continue
        api_files.append(str(p.relative_to(repo)))
        in_doc = False
        for i, ln in enumerate(txt.splitlines(), 1):
            if ln.count('"""') % 2:
                in_doc = not in_doc
                continue
            if in_doc or ln.strip().startswith("#"):
                continue
            if evalpat.search(ln):
                hits.append(f"{p.relative_to(repo)}:{i}  {ln.strip()[:90]}")
    if hits:
        return Result("5.3b", "API 로 평가 문항 처리", "FAIL",
                      "API 스크립트가 평가 파일을 참조", hits)
    return Result("5.3b", "API 로 평가 문항 처리", "PASS",
                  f"API 사용 파일 {len(api_files)}개 중 평가 문항 참조 0건 "
                  "(5.3a 학습데이터 구축 용도만)", api_files)


def c_10_1d_probing(repo: Path, data: Path) -> Result:
    """[10.1d] LB 정답을 읽거나 역추적하는 코드가 없는가.

    필터본에는 answer 컬럼이 아예 없다. 원본에는 있으므로 그 파일을 읽는 코드를 찾는다.
    """
    hits, clean = [], []
    for p in files(repo, ["src/**/*.py", "scripts/**/*.sh"]):
        txt = p.read_text(errors="ignore")
        for i, ln in enumerate(txt.splitlines(), 1):
            if "deep_chal_math_leaderboard.csv" not in ln or "filtered" in ln:
                continue
            s_ = ln.strip()
            if s_.startswith("#") or s_.startswith('"""'):
                continue
            # 같은 문장에서 answer 컬럼을 읽는지 본다. question 만 읽으면 dedup 용도다.
            ctx = "\n".join(txt.splitlines()[max(0, i - 3):i + 3])
            if re.search(r"[\"']answer[\"']|\.answer\b", ctx):
                hits.append(f"{p.relative_to(repo)}:{i}  {s_[:90]}")
            else:
                clean.append(f"{p.relative_to(repo)}:{i}  question 만 읽음 (dedup 용도)")
    lbf = data / "data/raw/deep_chal_math_leaderboard_filtered.csv"
    ev = []
    if lbf.exists():
        cols = next(csv.reader(lbf.open()))
        ev.append(f"배포 평가 파일 컬럼: {cols} (answer 없음)")
    if hits:
        return Result("10.1d", "LB probing", "MANUAL",
                      "answer 컬럼이 있는 LB 원본을 읽는 코드", ev + hits)
    return Result("10.1d", "LB probing", "PASS",
                  "LB 정답 컬럼을 읽는 코드 0건 — 원본을 여는 곳은 전부 dedup 금지목록 생성용",
                  ev + clean)


def c_f_4b_handlabel(repo: Path) -> Result:
    """[Foundational 4.b] 평가·검증 데이터의 수기 라벨링 정보를 제출물에 반영하지 않았는가."""
    return Result("F4.b", "수기 라벨 반영", "MANUAL",
                  "제출 답안은 모델 추론 산출물이어야 한다. 우리 라벨 감사 대상은 "
                  "train 과 자체 holdout 이며, LB/test 라벨은 만든 적도 쓴 적도 없다. "
                  "다만 자체 holdout 재라벨 14건이 모델 «선택»에 쓰였으므로 문서에 공개돼 있어야 한다.",
                  ["docs/methodology.md 의 val 편향 절",
                   "docs/val-bias-2026-08-26.md (작업 저장소)"])


# ══════════════════════════════════════════════════════════════════════
# 7·8. 제출물
# ══════════════════════════════════════════════════════════════════════

def c_8_1_submission(repo: Path, data: Path) -> Result:
    """[8.1/7.2] 제출 CSV 형식 — 컬럼·정수·빈값·행수."""
    subs = sorted((data / "submissions").glob("sub-*.csv"))
    if not subs:
        return Result("8.1", "제출 CSV 형식", "SKIP", "제출물 없음")
    latest = subs[-1]
    with latest.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
        cols = list(rows[0].keys()) if rows else []
    bad_int = [r for r in rows if not re.fullmatch(r"-?\d+", (r.get("answer") or "").strip())]
    blanks = [r for r in rows if not (r.get("answer") or "").strip()]
    ids = [r.get("id") or r.get("ID") for r in rows]
    ev = [f"검사 대상: {latest.name}", f"컬럼: {cols}", f"행수: {len(rows)}",
          f"중복 id: {len(ids) - len(set(ids))}", f"비정수: {len(bad_int)}", f"빈값: {len(blanks)}"]
    if bad_int or blanks or len(ids) != len(set(ids)):
        return Result("8.1", "제출 CSV 형식", "FAIL", "형식 위반", ev)
    if cols and cols[0] != "id":
        return Result("8.1", "제출 CSV 형식", "WARN",
                      "규정 문언은 'ID' 지만 Kaggle 채점기는 소문자 'id' 를 요구한다 "
                      "(8/23 실측). 최종 제출은 구글 폼이므로 당일 안내를 따를 것", ev)
    return Result("8.1", "제출 CSV 형식", "PASS", "정수·무결·중복없음", ev)


def c_8_2a_deliverables(repo: Path) -> Result:
    """[8.2a] 수상 후보자 제출물 6항목이 저장소에 있는가."""
    need = {
        "학습 코드": ["src/train/sft_full.py", "src/train/sft_lora.py"],
        "추론 코드": ["src/inference/generate.py", "src/inference/make_submission.py",
                   "scripts/run_dday_cascade.sh"],
        "데이터셋 목록·접근방법": ["data/external/DATA_SOURCES.md"],
        "재현 환경 설명": ["requirements-gpu.txt", "requirements-local.txt", "REPRODUCE.md"],
        "방법론 문서": ["docs/methodology.md"],
        "실행 방법 README (8/28 공지)": ["README.md"],
    }
    missing = []
    for k, ps in need.items():
        for p in ps:
            if not (repo / p).exists():
                missing.append(f"{k}: {p}")
    # 가중치는 파일이 아니라 «접근 방법»으로 충족된다
    weights_ok = any("dlc2026-weights" in (repo / f).read_text(errors="ignore")
                     for f in ["README.md", "REPRODUCE.md"] if (repo / f).exists())
    if not weights_ok:
        missing.append("학습된 모델 가중치: 배포처 안내가 README/REPRODUCE 에 없음")
    if missing:
        return Result("8.2a", "제출물 6항목", "FAIL", f"{len(missing)}건 누락", missing)
    return Result("8.2a", "제출물 6항목", "PASS",
                  "학습코드·추론코드·가중치 배포처·데이터목록·환경·방법론 전부 존재")


def c_8_2b_reproducible(repo: Path) -> Result:
    """[8.2b] 재현 가능성 — 절대경로 하드코딩과 사설 자원 의존."""
    hard = re.compile(r"(?<![A-Za-z_{:-])/(workspace|Users/[a-z]+)/")
    hits = []
    for p in files(repo, ["src/**/*.py", "scripts/**/*.sh", "scripts/lib/*.sh"]):
        for i, ln in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            s = ln.strip()
            if s.startswith("#") or "${" in s or ":-" in s:
                continue
            if re.search(r"environ\.get\(|getenv\(|argparse|default=", s):
                continue
            if hard.search(ln):
                hits.append(f"{p.relative_to(repo)}:{i}  {s[:90]}")
    if hits:
        return Result("8.2b", "재현 가능성", "WARN",
                      f"절대경로 하드코딩 {len(hits)}건 — 채점관 환경에서 실행 불가", hits)
    return Result("8.2b", "재현 가능성", "PASS",
                  "실행 경로가 전부 환경변수로 매개화됨 (DLC_ROOT/DLC_CKPT/DLC_TMP)")


def c_8_2b_seeds(repo: Path) -> Result:
    """[8.2b] 결정성 — 시드 고정과 vLLM 배치 불변 설정."""
    envf = repo / "scripts/lib/vllm_env.sh"
    need = ["VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING",
            "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"]
    if not envf.exists():
        return Result("8.2b", "결정성", "WARN", "공용 환경 블록 없음")
    txt = envf.read_text()
    miss = [k for k in need if k not in txt]
    if miss:
        return Result("8.2b", "결정성", "WARN", "결정성 환경변수 누락: " + ", ".join(miss))
    seeded = sum(1 for p in files(repo, ["src/**/*.py"])
                 if re.search(r"seed", p.read_text(errors="ignore")))
    return Result("8.2b", "결정성", "PASS",
                  f"결정성 환경변수 4종 고정 · 시드 인자를 가진 스크립트 {seeded}개",
                  [f"{envf.relative_to(repo)}: " + ", ".join(need)])


def c_notice_train_filter(repo: Path, data: Path, train_sets: list[Path]) -> Result:
    """[8/03 공지] 운영진이 공개한 train 오류 627문항이 학습 데이터에 남아 있는가.

    공지 문언은 "학습 시 해당 문항을 제외하고 활용해 주시기 바랍니다" 로 **권고**이고
    금지 조항이 아니다. 그래도 채점관이 확인할 지점이므로 실제 잔존 수를 센다.
    """
    f = data / "data/raw/train_filtered_ids.csv"
    tf = data / "data/raw/deep_chal_math_train.csv"
    if not (f.exists() and tf.exists()):
        return Result("공지", "train 오류 627 제외", "SKIP", "목록 또는 train 원본 없음")
    with f.open(newline="") as fh:
        bad = {r["id"] for r in csv.DictReader(fh)}
    with tf.open(newline="") as fh:
        q2id = {norm_q(r["question"]): r["id"] for r in csv.DictReader(fh)}
    avail = [p for p in train_sets if p.exists()]
    if not avail:
        return Result("공지", "train 오류 627 제외", "SKIP", "학습 데이터 없음")
    ev, worst = [], 0
    for p in avail:
        hit = set()
        with p.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = next((m.get("content", "") for m in rec.get("messages", [])
                          if m.get("role") == "user"), "")
                i = q2id.get(norm_q(q))
                if i in bad:
                    hit.add(i)
        worst = max(worst, len(hit))
        ev.append(f"{p.name}: 오류문항 {len(hit)}개 포함 ({len(hit) / len(bad) * 100:.1f}% of {len(bad)})")
    refs = [str(x.relative_to(repo)) for x in files(repo, DATABUILD_GLOBS)
            if "train_filtered_ids" in x.read_text(errors="ignore")]
    ev.append("목록을 참조하는 빌더: " + (", ".join(refs) if refs else "없음"))
    if worst:
        return Result("공지", "train 오류 627 제외", "WARN",
                      f"최대 {worst}개 잔존. **공지는 권고이지 금지가 아니다**(규칙 4·5 에 조항 없음). "
                      "다만 목록이 8/03 공개인 반면 일부 학습 데이터가 그 이전 산출물이므로 "
                      "시점과 영향을 문서에 밝힐 것", ev)
    return Result("공지", "train 오류 627 제외", "PASS", "잔존 0개", ev)


def c_notice_public_repo(repo: Path) -> Result:
    """[8/28 공지] 제출 저장소 Public + README 요건."""
    rd = repo / "README.md"
    if not rd.exists():
        return Result("공지", "Public 저장소·README", "FAIL", "README.md 없음")
    txt = rd.read_text()
    need = {"실행 환경": r"requirements|CUDA|GPU|환경",
            "실행 방법": r"python |bash |실행",
            "베이스 모델 명시": r"Qwen2\.5-3B-Instruct",
            "가중치 접근처": r"huggingface|dlc2026-weights"}
    miss = [k for k, p in need.items() if not re.search(p, txt, re.I)]
    if miss:
        return Result("공지", "Public 저장소·README", "WARN",
                      "README 누락 항목: " + ", ".join(miss))
    return Result("공지", "Public 저장소·README", "MANUAL",
                  f"README 요건 충족 ({len(txt.splitlines())}줄). "
                  "**8/31 제출 시 저장소를 Public 으로 전환해야 한다 — 현재 Private.**")


def c_f_6ab_sharing(repo: Path) -> Result:
    """[Foundational 6.a/6.b vs 8/28 공지] 코드 공개 시점·장소의 긴장."""
    return Result("F6.b", "코드 공개 장소", "MANUAL",
                  "Kaggle Foundational 6.b 는 공개 공유를 «Kaggle 포럼/노트북에» 하라고 하고, "
                  "6.a 는 대회 기간 중 비공개(선별적) 공유를 금지한다. 운영진 8/28 공지는 "
                  "8/31 GitHub Public 을 요구한다. 8/31 은 모델 개발 마감(8/30) 이후이고 "
                  "**전 참가자에게 동일하게 요구된 공개**라 6.a 의 «선별적 공유» 에 해당하지 않는다. "
                  "운영진 지시가 대회별 규칙이므로 따르되, 마감 전 사전 공개는 하지 않는다.")


def c_weights_available(repo: Path) -> Result:
    """[8.2a/8.2b] 가중치를 채점관이 실제로 받을 수 있는가."""
    import urllib.error
    import urllib.request
    url = "https://huggingface.co/api/models/ahnjun0/dlc2026-weights"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            ok = r.status == 200
        return Result("8.2a", "가중치 공개 접근", "PASS" if ok else "WARN",
                      "익명 접근 가능", [url])
    except urllib.error.HTTPError as e:
        return Result("8.2a", "가중치 공개 접근", "WARN",
                      f"익명 접근 불가 (HTTP {e.code}) — 8/31 제출 시 Public 전환 필요", [url])
    except Exception as e:  # 네트워크 없음
        return Result("8.2a", "가중치 공개 접근", "SKIP", f"확인 불가: {type(e).__name__}")


# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="검사할 저장소 루트")
    ap.add_argument("--data-root", default=None, help="데이터·제출물이 있는 저장소 (기본: --repo)")
    ap.add_argument("--train-set", action="append", default=[],
                    help="배포 모델의 학습 데이터 jsonl (여러 번 지정)")
    ap.add_argument("--strict", action="store_true", help="WARN 도 실패로 취급")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    data = Path(a.data_root).resolve() if a.data_root else repo
    tsets = [Path(p) for p in a.train_set]

    results = [
        c_4_1a_base_model(repo), c_4_1b_merge(repo), c_4_2_techniques(repo),
        c_4_3_2_inference_network(repo), c_4_3_3_pretrain(repo),
        c_5_1b_contamination(repo, data, tsets), c_val_leak(repo, data, tsets),
        c_5_2c_sources(repo), c_5_3bc_api(repo), c_10_1d_probing(repo, data),
        c_f_4b_handlabel(repo),
        c_8_1_submission(repo, data), c_8_2a_deliverables(repo),
        c_8_2b_reproducible(repo), c_8_2b_seeds(repo), c_weights_available(repo),
        c_notice_train_filter(repo, data, tsets), c_notice_public_repo(repo), c_f_6ab_sharing(repo),
    ]

    mark = {"PASS": "PASS  ", "FAIL": "FAIL  ", "WARN": "WARN  ",
            "MANUAL": "MANUAL", "SKIP": "SKIP  "}
    print(f"\n대회 규정 검증기 — 대상 {repo}\n" + "=" * 78)
    for r in sorted(results, key=lambda x: (STATUS_ORDER[x.status], x.clause)):
        print(f"[{mark[r.status]}] {r.clause:<7} {r.name}")
        print(f"           {r.detail}")
        for e in r.evidence[:12]:
            print(f"             · {e}")
        if len(r.evidence) > 12:
            print(f"             · … 외 {len(r.evidence) - 12}건")
    tally = {k: sum(1 for r in results if r.status == k) for k in STATUS_ORDER}
    print("=" * 78)
    print("  ".join(f"{k} {v}" for k, v in tally.items() if v))
    fails = tally["FAIL"] + (tally["WARN"] if a.strict else 0)
    print(("결론: 기계 검사 통과 — MANUAL 항목은 사람이 확인할 것"
           if not fails else f"결론: 조치 필요 {fails}건"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

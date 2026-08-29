"""LoRA SFT 학습 (GPU 환경 전용).

설계 근거 (research/prior-work-report.md):
  - LoRA all-linear(MLP 포함) 고랭크 == full FT 근접 (LoRA Without Regret)
  - LR은 full FT의 ~10배 (1e-4)
  - short-CoT 데이터 + seq 4096 + packing

사용 (4090에서):
    python src/train/sft_lora.py --data data/processed/sft_v1.jsonl \
        --output <체크포인트 경로>/exp-003_sft-v1 [--rank 128] [--epochs 2]
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch (24GB에서 seq4096+r128은 1이 한계 — 어휘 152k 로짓 메모리)")
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument(
        "--attn", default=None,
        help="attention 구현. 예: kernels-community/flash-attn2. "
             "packing(padding-free)은 FlashAttention 계열에서만 샘플 간 attention이 "
             "올바르게 차단된다 — 미지정 시 TRL이 경고를 내며 교차 오염 가능 "
             "(2026-08-06 발견: v2 등 기존 SFT가 전부 이 상태였음)",
    )
    ap.add_argument(
        "--assistant-only-loss", action="store_true",
        help="assistant 토큰에만 loss (기본은 system/user 포함 전체 시퀀스)",
    )
    ap.add_argument("--no-packing", action="store_true", help="packing 비활성 (대조군용)")
    ap.add_argument("--resume", action="store_true",
                    help="output_dir 의 마지막 체크포인트에서 이어서 학습. "
                         "**컨테이너가 예고 없이 재시작된다**(8/24 14:44 실측 — 154/190 에서 소실). "
                         "긴 학습에는 --n-checkpoints 와 함께 반드시 켤 것")
    ap.add_argument("--neftune", type=float, default=None,
                    help="NEFTune 임베딩 잡음 α (arXiv:2310.05914). **엔트로피를 올리는 유일한 손잡이** — "
                         "axis-map §22 참조: maj@k 에서 출력 엔트로피는 잡음이 아니라 자산이다. "
                         "판정은 단독 점수가 아니라 **φ(오류상관)** 로 한다")
    ap.add_argument("--n-checkpoints", type=int, default=0,
                    help=">0이면 **등간격으로 N개 저장**한다 (체크포인트 평균용). "
                         "종전 save_strategy='epoch'은 2에폭에서 2개뿐이라 "
                         "NVIDIA/AIMO-2가 쓰는 '마지막 4개 평균'을 재현할 수 없다")
    ap.add_argument("--pause-token-id", type=int, default=-1,
                    help="MBP: 이 토큰 위치의 label을 -100으로 마스킹 (문맥에는 남긴다). "
                         "-1이면 비활성. Qwen2.5는 <|fim_pad|>=151662 등 미사용 특수토큰 재활용")
    ap.add_argument("--init-adapter", default=None,
                    help="이 LoRA 어댑터를 **초기값으로 이어서** 학습한다 (없으면 새로 시작). "
                         "왜 필요한가: LoRA 델타는 A의 무작위 부분공간에 갇히므로 "
                         "독립 학습 둘은 **구조적으로 직교**한다(주각 코사인 0.093 = 무작위 0.092). "
                         "공통 어댑터에서 분기하면 부분공간이 공유돼 τ_new − τ_init 이 "
                         "**의미 있는 방향**이 되고 λ 스케일이 진짜 강도 조절이 된다 "
                         "(Linear Mode Connectivity, arXiv:1912.05671).")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=args.data, split="train")

    if args.init_adapter:
        # 기존 어댑터를 **초기값으로** 올려 이어 학습한다 (peft_config 는 주지 않는다).
        # is_trainable=True 여야 그 어댑터가 실제로 갱신된다.
        from transformers import AutoModelForCausalLM
        _base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
        model_or_id = PeftModel.from_pretrained(_base, args.init_adapter, is_trainable=True)
        peft_cfg = None
        print(f"초기값 어댑터 = {args.init_adapter} (부분공간을 공유해 이어 학습)")
    else:
        model_or_id = args.model
        peft_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.0,
            target_modules="all-linear",  # MLP 포함 필수 (attention-only 금지)
            task_type="CAUSAL_LM",
        )

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.seq_len,
        packing=not args.no_packing,
        **({"neftune_noise_alpha": args.neftune} if args.neftune else {}),
        assistant_only_loss=args.assistant_only_loss,
        **({"model_init_kwargs": {"attn_implementation": args.attn}} if args.attn else {}),
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        logging_steps=10,
        # epoch 경계마다 저장한다. 종전 설정(save_steps=500)은 총 스텝이 190뿐이라
        # **한 번도 발동하지 않았고**, 그 결과 v2~v5a 전 모델의 epoch-1이 존재하지
        # 않았다 (2026-08-06 발견). LoRA 어댑터는 ~470MB라 2개 보관도 부담 없다.
        **({"save_strategy": "steps", "save_steps": 1.0/args.n_checkpoints,
             "save_total_limit": args.n_checkpoints}
           if args.n_checkpoints > 0 else
           {"save_strategy": "epoch",
            "save_total_limit": args.epochs if isinstance(args.epochs, int) else 3}),
        seed=args.seed,
        report_to="wandb" if args.wandb_run else "none",
        run_name=args.wandb_run,
        dataset_num_proc=8,
    )

    trainer = SFTTrainer(
        model=model_or_id,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )

    if args.pause_token_id >= 0:
        # **MBP(Masked Boundary Pause)의 핵심**: pause 토큰은 **문맥에는 남고 label에서만 빠진다.**
        #
        # 논문(Towards Understanding Pause Token Fine-Tuning Dynamics, ACL 익명 투고) 초록:
        #   "pause tokens placed at reasoning-step boundaries **with their loss masked**"
        #   "effects are present even when pause tokens are **absent at inference**
        #    and never appear as supervised targets"
        # 즉 pause를 정답 토큰처럼 예측시키면 실험이 아예 달라진다. 마스킹이 필수 조건이다.
        base_collator = trainer.data_collator
        pid = args.pause_token_id

        def masked_collator(features):
            batch = base_collator(features)
            if "labels" in batch and "input_ids" in batch:
                batch["labels"] = batch["labels"].masked_fill(batch["input_ids"] == pid, -100)
            return batch

        trainer.data_collator = masked_collator
        print(f"[MBP] pause_token_id={pid} 위치의 label을 -100으로 마스킹")

    torch.manual_seed(args.seed)
    _ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        _ckpt = get_last_checkpoint(args.output)
        print(f"[resume] 이어받을 체크포인트: {_ckpt or '없음 — 처음부터'}", flush=True)
    trainer.train(resume_from_checkpoint=_ckpt)
    trainer.save_model(args.output + "/final")
    print("saved:", args.output + "/final")


if __name__ == "__main__":
    main()

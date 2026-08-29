"""Full FT (문샷 long-CoT 전용) — LoRA 없이 전 파라미터 학습.

메모리 전략 (4090 48GB 전제):
  - bf16 + adamw_8bit (옵티마이저 ~12GB) + gradient checkpointing
  - seq 8192 (moonshot_r1 p90=6.6k토큰 수납), packing
  - batch 1 × accum 32

사용: python src/train/sft_full.py --data data/processed/moonshot_r1.jsonl \
        --output <체크포인트 경로>/exp-015_moonshot [--smoke]
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="3 step 메모리 스모크")
    ap.add_argument("--liger", action="store_true", help="Liger 융합 커널 (로짓 메모리 절감 — 32GB 카드용)")
    ap.add_argument("--paged-optim", action="store_true", help="paged_adamw_8bit (옵티마이저 CPU 페이징 — 32GB 카드용)")
    ap.add_argument(
        "--attn", default=None,
        help="attention 구현. 예: kernels-community/flash-attn2. "
             "packing(padding-free)은 FlashAttention 계열에서만 샘플 간 attention이 "
             "올바르게 차단된다. **LoRA 판과 비교하려면 반드시 동일하게 지정할 것**",
    )
    ap.add_argument("--epoch-save", action="store_true",
                    help="epoch 경계마다 저장 (기본은 save_steps). epoch 비교가 필요할 때")
    ap.add_argument("--save-limit", type=int, default=1,
                    help="보관할 체크포인트 수. **기본 1** — 3B full FT의 fp32 체크포인트는 개당 "
                         "12GB라 2개면 24GB가 필요하다. 2026-08-08에 이 값이 2여서 190/190 스텝을 "
                         "완주하고도 저장에서 `No space left on device`로 죽었다. "
                         "epoch 비교가 꼭 필요할 때만 2로 올리고, 디스크를 먼저 확인할 것")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=args.data, split="train")

    cfg = SFTConfig(
        output_dir=args.output,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else -1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.seq_len,
        packing=True,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.paged_optim else "adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if args.smoke else 10,
        save_strategy="no" if args.smoke else ("epoch" if args.epoch_save else "steps"),
        save_steps=args.save_steps,
        save_total_limit=args.save_limit,
        save_only_model=True,  # 디스크 50GB 제약: 옵티마이저 제외 (재개 불가 대신 공간 확보)
        seed=args.seed,
        report_to=[],
        use_liger_kernel=args.liger,
        **({"model_init_kwargs": {"attn_implementation": args.attn}} if args.attn else {}),
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds)
    trainer.train()
    if args.smoke:
        print(f"SMOKE_OK peak_vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB")
    else:
        # 디스크 사고 방지 (8/5 실측 교훈): 저장 전 체크포인트 정리 + bf16 저장(6GB)
        import glob as _g
        import shutil as _sh
        for d in _g.glob(f"{args.output}/checkpoint-*"):
            _sh.rmtree(d, ignore_errors=True)
        trainer.model.to(torch.bfloat16)
        trainer.save_model(f"{args.output}/final")
        print(f"saved: {args.output}/final")


if __name__ == "__main__":
    main()

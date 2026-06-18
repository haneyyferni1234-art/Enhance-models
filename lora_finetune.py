"""
LoRA Fine-Tuning for FusionModelWrapper
========================================
Trains the expanded embedding vectors + attention layers on fused text
so the model learns to naturally predict fused tokens.

Usage:
    # Default: 200 steps, batch_size=2, on CPU
    python3 lora_finetune.py

    # GPU with more steps
    python3 lora_finetune.py --steps 500 --batch-size 4 --lr 2e-4

    # Resume from checkpoint
    python3 lora_finetune.py --load ./fused-qwen --resume ./lora-checkpoint
"""

import json, math, os, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


# ── Import our fusion wrapper ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from fusion_wrapper import from_pretrained, FusionModelWrapper


# ─── Fused Text Dataset ──────────────────────────────────────────────────────

class FusedTextDataset(Dataset):
    """Encode → fuse corpus documents on-the-fly for causal LM training."""

    def __init__(self, corpus_path, wrapper, max_samples=5000, seq_len=1024):
        self.wrapper = wrapper
        self.seq_len = seq_len
        self.samples = []

        print(f"  [LoRA] Building fused dataset from {corpus_path}...")
        with open(corpus_path) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                text = json.loads(line)["text"]
                # Fuse the text
                fused_ids = wrapper.encode(text)
                # Truncate/pad to seq_len
                if len(fused_ids) > seq_len:
                    fused_ids = fused_ids[:seq_len]
                if len(fused_ids) < 2:
                    continue
                self.samples.append(fused_ids)

        print(f"  [LoRA] Dataset: {len(self.samples):,} samples, seq_len={seq_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch, pad_id):
    """Pad batch to equal length. Labels = input_ids (shifted inside model)."""
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ─── Training Loop ───────────────────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("  FusionModelWrapper — LoRA Fine-Tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # 1. Load or create fused model
    if args.load:
        from fusion_wrapper import from_fused_pretrained
        print(f"\n[1] Loading fused model from {args.load}...")
        wrapper, tokenizer = from_fused_pretrained(
            args.load, device_map=device, torch_dtype=torch.float32
        )
    else:
        print(f"\n[1] Creating fused model from {args.model}...")
        wrapper, tokenizer = from_pretrained(
            args.model, device_map=device, torch_dtype=torch.float32
        )

    model = wrapper.model
    pad_id = tokenizer.pad_token_id or 0

    # 2. Add LoRA
    print(f"\n[2] Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )

    # Prepare model for LoRA (FP32/FP16 — not quantized)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.train()

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # 3. Build dataset + dataloader
    print(f"\n[3] Preparing dataset...")
    dataset = FusedTextDataset(
        args.corpus, wrapper,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
    )

    # 4. Optimizer
    print(f"\n[4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    # 5. Training loop
    print(f"\n[5] Training for {args.steps} steps...")
    global_step = 0
    best_loss = float("inf")
    log_interval = max(1, args.steps // 20)

    model.zero_grad()
    while global_step < args.steps:
        for batch in dataloader:
            if global_step >= args.steps:
                break

            batch = batch.to(device)

            outputs = model(
                input_ids=batch,
                labels=batch,  # causal LM: predict next token
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_interval == 0 or global_step == args.steps:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Step {global_step:>4}/{args.steps} | loss: {loss.item():.4f} | lr: {lr_now:.2e}")

                if loss.item() < best_loss:
                    best_loss = loss.item()

    # 6. Save LoRA weights + full model
    if args.output:
        print(f"\n[6] Saving to {args.output}...")
        os.makedirs(args.output, exist_ok=True)

        # Merge LoRA into base model and save full fused model
        model = model.merge_and_unload()
        wrapper.model = model
        wrapper.save_pretrained(args.output)

        # Also save LoRA adapter separately
        lora_path = os.path.join(args.output, "lora")
        model.save_pretrained(lora_path)
        print(f"  LoRA adapter saved to {lora_path}")

    print(f"\n{'='*60}")
    print(f"  Done! Best loss: {best_loss:.4f}")
    print(f"  Fused model: {args.output}")
    print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for FusionModelWrapper")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--load", default=None, help="Load existing fused model instead of creating new")
    parser.add_argument("--corpus", default="./accepted-001.jsonl")
    parser.add_argument("--output", default="./fused-lora")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024, help="Max fused tokens per sample")
    parser.add_argument("--max-samples", type=int, default=1000, help="Docs to sample from corpus")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import DataConfig, OptimConfig, SupervisedConfig
from .data import RerankDataset, collate_ranknet
from .modeling import build_model, load_tokenizer
from .metrics import compute_metrics
from .utils import console, get_device


def parse_args() -> Tuple[DataConfig, OptimConfig, SupervisedConfig]:
    parser = argparse.ArgumentParser(description="Supervised RankNet training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="nomic-ai/modernbert-base")
    parser.add_argument("--output-dir", type=str, default="outputs/sft")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--relevant-top-k", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=400)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        max_candidates=args.max_candidates,
        max_length=args.max_length,
        sample_frac=args.sample_frac,
        relevant_top_k=args.relevant_top_k,
    )
    optim_cfg = OptimConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_epochs,
    )
    sup_cfg = SupervisedConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    return data_cfg, optim_cfg, sup_cfg


def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    device = scores.device
    unique_groups = torch.unique(group_ids)
    total_loss = torch.zeros(1, device=device)
    pair_count = torch.zeros(1, device=device)
    for g in unique_groups:
        mask = group_ids == g
        g_scores = scores[mask]
        g_labels = labels[mask]
        if g_scores.numel() < 2:
            continue
        score_diff = g_scores.unsqueeze(0) - g_scores.unsqueeze(1)
        label_diff = g_labels.unsqueeze(0) - g_labels.unsqueeze(1)
        pair_mask = label_diff > 0
        if pair_mask.any():
            selected = score_diff[pair_mask]
            total_loss += torch.nn.functional.softplus(-selected).sum()
            pair_count += pair_mask.sum()
    if pair_count.item() == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / pair_count


def evaluate(model, tokenizer, dataset: RerankDataset, data_cfg: DataConfig, device: torch.device):
    model.eval()
    predicted, target = [], []
    with torch.no_grad():
        for sample in dataset:
            candidates = sample["candidates"]
            encoded = tokenizer(
                [sample["query"]] * len(candidates),
                candidates,
                padding=True,
                truncation=True,
                max_length=data_cfg.max_length,
                return_tensors="pt",
            ).to(device)
            scores = model(encoded["input_ids"], encoded["attention_mask"]).cpu()
            ordering = torch.argsort(scores, descending=True).tolist()
            predicted.append(ordering)
            target.append(sample["oracle_ranking"][: data_cfg.relevant_top_k])
    return compute_metrics(target, predicted, k=min(10, data_cfg.max_candidates))


def main() -> None:
    data_cfg, optim_cfg, sup_cfg = parse_args()
    device = get_device()
    console.print(f"Using device: {device}")

    tokenizer = load_tokenizer(sup_cfg.model_id)
    train_dataset = RerankDataset(
        Path(data_cfg.data_dir) / data_cfg.train_file,
        max_candidates=data_cfg.max_candidates,
        sample_frac=data_cfg.sample_frac,
    )
    val_dataset = RerankDataset(
        Path(data_cfg.data_dir) / data_cfg.val_file,
        max_candidates=data_cfg.max_candidates,
        sample_frac=data_cfg.sample_frac,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=sup_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_ranknet(tokenizer, data_cfg.max_length),
    )

    model = build_model(
        model_id=sup_cfg.model_id,
        lora_r=sup_cfg.lora_r,
        lora_alpha=sup_cfg.lora_alpha,
        lora_dropout=sup_cfg.lora_dropout,
        lora_target_modules=tuple(sup_cfg.lora_target_modules),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg.learning_rate, weight_decay=optim_cfg.weight_decay)
    num_training_steps = len(train_loader) * optim_cfg.num_train_epochs
    warmup_steps = int(num_training_steps * optim_cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    global_step = 0
    best_metric = -float("inf")
    sup_cfg.output_dir = str(Path(sup_cfg.output_dir))
    Path(sup_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(optim_cfg.num_train_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            group_ids = batch.group_ids.to(device)

            scores = model(input_ids, attention_mask)
            loss = ranknet_loss(scores, labels, group_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if global_step % sup_cfg.logging_steps == 0:
                console.log(f"step={global_step} loss={loss.item():.4f}")

            if global_step % sup_cfg.eval_steps == 0 and global_step > 0:
                metrics = evaluate(model, tokenizer, val_dataset, data_cfg, device)
                console.log(f"eval@{global_step} {json.dumps(metrics)}")
                metric_score = metrics.get("ndcg@10", 0.0)
                if metric_score > best_metric:
                    best_metric = metric_score
                    save_path = Path(sup_cfg.output_dir) / "best"
                    save_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(save_path))
                    tokenizer.save_pretrained(save_path)
                    console.log(f"saved best model to {save_path}")

            if global_step % sup_cfg.save_steps == 0 and global_step > 0:
                checkpoint_dir = Path(sup_cfg.output_dir) / f"step-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(checkpoint_dir))
                tokenizer.save_pretrained(checkpoint_dir)
            global_step += 1

        metrics = evaluate(model, tokenizer, val_dataset, data_cfg, device)
        console.log(f"epoch={epoch} {json.dumps(metrics)}")

    final_dir = Path(sup_cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    console.print(f"Training complete. Final model at {final_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .config import DataConfig, PPOConfig
from .data import RerankDataset
from .metrics import ndcg_at_k
from .modeling import RerankerModel, load_tokenizer
from .utils import console, get_device


def parse_args() -> Tuple[DataConfig, PPOConfig]:
    parser = argparse.ArgumentParser(description="Policy-gradient fine-tuning for reranker")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sft-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/ppo")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--rollout-samples", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.0)
    parser.add_argument("--kl-penalty", type=float, default=0.02)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--relevant-top-k", type=int, default=10)
    args = parser.parse_args()

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        max_length=args.max_length,
        max_candidates=args.max_candidates,
        relevant_top_k=args.relevant_top_k,
    )
    ppo_cfg = PPOConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.clip_range,
        value_loss_coef=args.value_loss_coef,
        kl_penalty=args.kl_penalty,
        target_kl=args.target_kl,
        gamma=args.gamma,
        lam=args.lam,
        rollout_samples=args.rollout_samples,
        temperature=args.temperature,
    )
    return data_cfg, ppo_cfg, args.epochs, args.seed


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_ranking(scores: torch.Tensor, temperature: float) -> Tuple[list[int], torch.Tensor, torch.Tensor]:
    remaining = list(range(scores.shape[0]))
    ordering = []
    log_probs = []
    entropies = []
    for _ in range(len(remaining)):
        logits = scores[remaining] / temperature
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        ordering.append(remaining[action.item()])
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        remaining.pop(action.item())
    log_prob_tensor = torch.stack(log_probs).sum()
    entropy_tensor = torch.stack(entropies).sum()
    return ordering, log_prob_tensor, entropy_tensor


def compute_reward(target_order: list[int], predicted_order: list[int], k: int) -> float:
    return ndcg_at_k(target_order, predicted_order, k)


def main():
    data_cfg, ppo_cfg, num_epochs, seed = parse_args()
    console.log("loading resources")
    set_seed(seed)
    device = get_device()

    sft_checkpoint_path = Path(ppo_cfg.sft_checkpoint)
    # The 'best' checkpoint is only saved if the metric improves. Fallback to 'final'.
    if sft_checkpoint_path.name == "best" and not sft_checkpoint_path.exists():
        sft_checkpoint_path = sft_checkpoint_path.parent / "final"
        console.print(f"'best' checkpoint not found, falling back to '{sft_checkpoint_path}'")

    tokenizer = load_tokenizer(str(sft_checkpoint_path))
    model = RerankerModel.from_pretrained(str(sft_checkpoint_path))
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=ppo_cfg.learning_rate)

    dataset = RerankDataset(
        Path(data_cfg.data_dir) / data_cfg.train_file,
        max_candidates=data_cfg.max_candidates,
    )

    def collate_identity(batch):
        return batch

    data_loader = DataLoader(dataset, batch_size=ppo_cfg.batch_size, shuffle=True, collate_fn=collate_identity)

    baseline = 0.0
    step = 0
    ppo_cfg.output_dir = str(Path(ppo_cfg.output_dir))
    Path(ppo_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            total_loss = 0.0
            total_logprob = 0.0
            total_reward = 0.0
            total_entropy = 0.0
            for sample in batch:
                candidates = sample["candidates"]
                query = sample["query"]
                if not candidates:
                    continue
                encoded = tokenizer(
                    [query] * len(candidates),
                    candidates,
                    padding=True,
                    truncation=True,
                    max_length=data_cfg.max_length,
                    return_tensors="pt",
                ).to(device)
                scores = model(encoded["input_ids"], encoded["attention_mask"])
                ordering, log_prob, entropy = sample_ranking(scores, ppo_cfg.temperature)
                reward = compute_reward(sample["oracle_ranking"], ordering, data_cfg.relevant_top_k)
                baseline = 0.9 * baseline + 0.1 * reward
                advantage = reward - baseline
                loss = -advantage * log_prob - 0.01 * entropy
                total_loss += loss
                total_reward += reward
                total_logprob += log_prob.item()
                total_entropy += entropy.item()

            if isinstance(total_loss, float):
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                console.log(
                    f"epoch={epoch} step={step} loss={total_loss.item():.4f} reward={total_reward / max(1,len(batch)):.4f}"
                )
            step += 1

        checkpoint_dir = Path(ppo_cfg.output_dir) / f"epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(checkpoint_dir)
        console.log(f"saved checkpoint to {checkpoint_dir}")

    final_dir = Path(ppo_cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    console.print(f"RL fine-tuning complete. Final model at {final_dir}")


if __name__ == "__main__":
    main()

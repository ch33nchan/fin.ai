from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import DataConfig, EvalConfig
from .data import RerankDataset
from .metrics import compute_metrics
from .modeling import RerankerModel, load_tokenizer
from .utils import console, get_device


def parse_args() -> tuple[DataConfig, EvalConfig]:
    parser = argparse.ArgumentParser(description="Evaluate reranker checkpoints")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--relevant-top-k", type=int, default=10)
    args = parser.parse_args()

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        max_length=args.max_length,
        max_candidates=args.max_candidates,
        relevant_top_k=args.relevant_top_k,
    )
    eval_cfg = EvalConfig(
        checkpoint=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
    )
    return data_cfg, eval_cfg


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    data_cfg, eval_cfg = parse_args()
    device = get_device()
    console.print(f"Evaluating on device: {device}")

    tokenizer = load_tokenizer(eval_cfg.checkpoint)
    model = RerankerModel.from_pretrained(eval_cfg.checkpoint)
    model.to(device)
    model.eval()

    dataset_path = Path(data_cfg.data_dir) / f"{eval_cfg.split}.jsonl"
    dataset = RerankDataset(dataset_path, max_candidates=data_cfg.max_candidates)
    if eval_cfg.max_eval_samples is not None:
        dataset.samples = dataset.samples[: eval_cfg.max_eval_samples]

    predicted, target = [], []
    with torch.no_grad():
        for sample_batch in batched(dataset, eval_cfg.batch_size):
            queries = []
            passages = []
            lengths = []
            for sample in sample_batch:
                counts = len(sample["candidates"])
                queries.extend([sample["query"]] * counts)
                passages.extend(sample["candidates"])
                lengths.append(counts)
            encoded = tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=data_cfg.max_length,
                return_tensors="pt",
            ).to(device)
            scores = model(encoded["input_ids"], encoded["attention_mask"]).cpu()
            offset = 0
            for sample, count in zip(sample_batch, lengths):
                candidate_scores = scores[offset : offset + count]
                ordering = torch.argsort(candidate_scores, descending=True).tolist()
                predicted.append(ordering)
                target.append(sample["oracle_ranking"][: data_cfg.relevant_top_k])
                offset += count

    metrics = compute_metrics(target, predicted, k=min(10, data_cfg.max_candidates))
    console.print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

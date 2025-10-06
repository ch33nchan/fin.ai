import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class RerankSample(Dict):
    query: str
    candidates: List[str]
    oracle_ranking: List[int]


class RerankDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        max_candidates: int = 40,
        sample_frac: Optional[float] = None,
    ) -> None:
        self.samples: List[RerankSample] = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                candidates = record["candidates"][:max_candidates]
                oracle = record.get("oracle_ranking", list(range(len(candidates))))
                oracle = [idx for idx in oracle if idx < len(candidates)]
                self.samples.append(
                    {
                        "query": record["query"],
                        "candidates": candidates,
                        "oracle_ranking": oracle,
                    }
                )
        if sample_frac is not None:
            cutoff = max(1, int(len(self.samples) * sample_frac))
            self.samples = self.samples[:cutoff]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RerankSample:
        return self.samples[idx]


class RankNetBatch:
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.group_ids = group_ids


def build_ranknet_pairs(batch: List[RerankSample], tokenizer, max_length: int):
    queries, passages, labels, group_ids = [], [], [], []
    for sample_idx, sample in enumerate(batch):
        query = sample["query"]
        for idx, passage in enumerate(sample["candidates"]):
            queries.append(query)
            passages.append(passage)
            # convert oracle ranking into relative score (lower rank -> higher score)
            try:
                rank = sample["oracle_ranking"].index(idx)
                score = 1.0 - rank / max(1, len(sample["oracle_ranking"]) - 1)
            except ValueError:
                score = 0.0
            labels.append(score)
            group_ids.append(sample_idx)

    encoded = tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return RankNetBatch(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(labels, dtype=torch.float),
        torch.tensor(group_ids, dtype=torch.long),
    )


def collate_ranknet(tokenizer, max_length: int):
    def _fn(batch: List[RerankSample]):
        return build_ranknet_pairs(batch, tokenizer, max_length)

    return _fn

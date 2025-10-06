from typing import Dict, List

import torch


def average_precision(y_true: List[int], y_pred: List[int]) -> float:
    hits = 0
    sum_precisions = 0.0
    for idx, pred in enumerate(y_pred, start=1):
        if pred in y_true:
            hits += 1
            sum_precisions += hits / idx
    if not y_true:
        return 0.0
    return sum_precisions / len(y_true)


def mean_average_precision(seqs_true: List[List[int]], seqs_pred: List[List[int]]) -> float:
    scores = [average_precision(t, p) for t, p in zip(seqs_true, seqs_pred)]
    return sum(scores) / max(1, len(scores))


def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int = 10) -> float:
    dcg = 0.0
    idcg = 0.0
    for idx in range(k):
        if idx < len(y_pred) and y_pred[idx] in y_true:
            dcg += 1.0 / torch.log2(torch.tensor(idx + 2.0)).item()
        if idx < len(y_true):
            idcg += 1.0 / torch.log2(torch.tensor(idx + 2.0)).item()
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(y_true: List[int], y_pred: List[int], k: int = 10) -> float:
    if not y_true:
        return 0.0
    hits = sum(1 for item in y_pred[:k] if item in y_true)
    return hits / len(y_true)


def kendall_tau(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            true_order = (y_true[i] - y_true[j])
            pred_order = (y_pred.index(y_true[i]) - y_pred.index(y_true[j])) if y_true[i] in y_pred and y_true[j] in y_pred else 0
            concordant += 1 if true_order * pred_order > 0 else 0
            discordant += 1 if true_order * pred_order < 0 else 0
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def compute_metrics(target_lists: List[List[int]], predicted_lists: List[List[int]], k: int = 10) -> Dict[str, float]:
    return {
        "map": mean_average_precision(target_lists, predicted_lists),
        "ndcg@{}".format(k): sum(ndcg_at_k(t, p, k) for t, p in zip(target_lists, predicted_lists)) / max(1, len(target_lists)),
        "recall@{}".format(k): sum(recall_at_k(t, p, k) for t, p in zip(target_lists, predicted_lists)) / max(1, len(target_lists)),
        "kendall_tau": sum(kendall_tau(t, p) for t, p in zip(target_lists, predicted_lists)) / max(1, len(target_lists)),
    }

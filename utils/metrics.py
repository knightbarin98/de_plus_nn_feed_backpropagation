from __future__ import annotations

import torch


def per_class_metrics(cm: torch.Tensor) -> dict:
    cm = cm.to(torch.float64)
    tp = torch.diagonal(cm)          # (C×C) -> diagonal (C,): aciertos por clase
    row = cm.sum(dim=1)              # (C×C) sumando columnas -> (C,): total por clase REAL
    col = cm.sum(dim=0)              # (C×C) sumando filas    -> (C,): total por clase PREDICHA
    recall = torch.where(row > 0, tp / row.clamp(min=1), torch.zeros_like(tp))        # (C,) / (C,) -> (C,)
    precision = torch.where(col > 0, tp / col.clamp(min=1), torch.zeros_like(tp))     # (C,) / (C,) -> (C,)
    denom = precision + recall                                                        # (C,)
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-300), torch.zeros_like(tp))  # (C,)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist()}


def classification_metrics(cm: torch.Tensor) -> dict:
    total = float(cm.sum())
    pc = per_class_metrics(cm)
    n = len(pc["precision"])
    return {
        "accuracy": float(torch.diagonal(cm).sum()) / total if total else 0.0,    # traza (C×C) / N
        "precision_per_class": pc["precision"],
        "recall_per_class": pc["recall"],
        "f1_per_class": pc["f1"],
        "macro_precision": sum(pc["precision"]) / n,
        "macro_recall": sum(pc["recall"]) / n,
        "macro_f1": sum(pc["f1"]) / n,
    }


def class_counts(y: torch.Tensor, n_classes: int) -> list:
    return [int((y == k).sum()) for k in range(n_classes)]

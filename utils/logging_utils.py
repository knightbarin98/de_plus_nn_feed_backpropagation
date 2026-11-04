from __future__ import annotations

import csv
import gzip
import json
import math
import os

import torch

from config import ITERATION_NAME, ITERATION_START

FLOAT_FMT = "{:.10g}"
EPOCH_COLUMNS = [ITERATION_NAME, "train_loss", "test_loss"]


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else FLOAT_FMT.format(v)
    return v


def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_csv(path: str, header: list, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([_fmt(v) for v in row])


def write_epochs_csv(path: str, losses: list, test_losses: list) -> None:
    # backprop: la época 1 es el loss de los pesos iniciales (antes de la primera actualización)
    # DE: la generación 0 es el mejor individuo de la población inicial
    write_csv(path, EPOCH_COLUMNS, [[epoch, loss, test_loss] for epoch, (loss, test_loss)
                                    in enumerate(zip(losses, test_losses), start=ITERATION_START)])


def write_confusion_csv(path: str, cm: torch.Tensor, class_names: list) -> None:
    rows = [[f"real_{class_names[i]}"] + [int(v) for v in cm[i].tolist()] for i in range(cm.shape[0])]
    write_csv(path, ["real \\ predicha"] + [f"pred_{c}" for c in class_names], rows)


class GlobalEpochsWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = gzip.open(path, "wt", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["experiment_id", "subexperiment_id", "train_set", "test_set"] + EPOCH_COLUMNS)
        self.rows_written = 0

    def append(self, experiment_id: int, subexperiment_id: int, train_set: str, test_set: str,
               losses: list, test_losses: list) -> None:
        for epoch, (loss, test_loss) in enumerate(zip(losses, test_losses), start=ITERATION_START):
            self._w.writerow([experiment_id, subexperiment_id, train_set, test_set, epoch, _fmt(loss), _fmt(test_loss)])
        self.rows_written += len(losses)

    def close(self) -> None:
        self._f.close()

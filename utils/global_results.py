from __future__ import annotations

import os
import shutil

import torch

import config
from config import CLASS_NAMES, FEATURE_NAMES, RESULTS_DIR
from network import MLPNetwork
from .logging_utils import GlobalEpochsWriter, write_confusion_csv, write_csv, write_epochs_csv, write_json
from .metrics import classification_metrics
from .model_io import save_model
from .results import SUBEXPERIMENTS, subexperiment_summary, subexperiment_title
from .statistics import describe
from .visualization import plot_global, plot_subexperiment


def split_curves(all_epoch_losses: list, sub_rows: list) -> list:
    # main.py concatena los losses de los 70 subexperimentos en el mismo orden que sub_rows:
    # se vuelven a cortar con epochs_executed de cada fila
    curves, start = [], 0
    for r in sub_rows:
        end = start + r["epochs_executed"]
        curves.append(all_epoch_losses[start:end])
        start = end
    assert start == len(all_epoch_losses), (start, len(all_epoch_losses))
    return curves


def extremes(items: list, key: str) -> dict:
    values = [it[key] for it in items]
    lo, hi = min(values), max(values)
    return {"min_value": lo, "min_items": [it for it in items if it[key] == lo],
            "max_value": hi, "max_items": [it for it in items if it[key] == hi]}


def global_statistics(cm_global: torch.Tensor, finals: list, cumulatives: list, all_epoch_losses: list,
                      sub_rows: list, exp_rows: list) -> dict:
    sub_light = [{k: r[k] for k in ("experiment_id", "subexperiment", "test_accuracy", "final_training_loss")}
                 for r in sub_rows]
    exp_light = [{"experiment_id": e["experiment_id"], "test_accuracy_aggregated": e["test_accuracy_aggregated"]}
                 for e in exp_rows]
    return {
        "subexperiments": {
            "test_accuracy": describe([r["test_accuracy"] for r in sub_rows]),
            "macro_precision": describe([r["macro_precision"] for r in sub_rows]),
            "final_training_loss": describe(finals),
            "cumulative_training_loss": describe(cumulatives),
            "epochs_executed": describe([r["epochs_executed"] for r in sub_rows]),
            "extremes_test_accuracy": extremes(sub_light, "test_accuracy"),
            "converged_count": sum(r["converged"] for r in sub_rows),
        },
        "experiments": {
            "test_accuracy_aggregated": describe([e["test_accuracy_aggregated"] for e in exp_rows]),
            "extremes": extremes(exp_light, "test_accuracy_aggregated"),
        },
        "all_epochs_training_loss": describe(all_epoch_losses),
        "total_epoch_rows": len(all_epoch_losses),
        "global_confusion_matrix_test": cm_global,
        "global_test_accuracy_from_confusion": float(torch.diagonal(cm_global).sum()) / float(cm_global.sum()),
        "global_metrics_test": classification_metrics(cm_global),
    }


def save_global(cm_global: torch.Tensor, finals: list, cumulatives: list, all_epoch_losses: list,
                sub_rows: list, exp_rows: list) -> dict:
    global_dir = os.path.join(RESULTS_DIR, "global")
    os.makedirs(global_dir, exist_ok=True)
    write_json(os.path.join(global_dir, "config.json"), config.to_dict())

    # todas las épocas de los 70 subexperimentos en un solo archivo comprimido
    curves = split_curves(all_epoch_losses, sub_rows)
    writer = GlobalEpochsWriter(os.path.join(global_dir, "all_epochs.csv.gz"))
    for r, losses in zip(sub_rows, curves):
        train_set, test_set = r["subexperiment"].split("_to_")
        writer.append(r["experiment_id"], r["subexperiment_id"], train_set, test_set, losses)
    writer.close()

    write_csv(os.path.join(global_dir, "subexperiments.csv"), list(sub_rows[0]), [list(r.values()) for r in sub_rows])
    write_csv(os.path.join(global_dir, "experiments.csv"), list(exp_rows[0]), [list(e.values()) for e in exp_rows])

    # matriz de confusión global: suma de los 70 subexperimentos
    write_confusion_csv(os.path.join(global_dir, "confusion_matrix_test_global.csv"), cm_global, CLASS_NAMES)

    stats = global_statistics(cm_global, finals, cumulatives, all_epoch_losses, sub_rows, exp_rows)
    plot_global(sub_rows, exp_rows, cm_global, curves, os.path.join(global_dir, "plots"))
    write_json(os.path.join(global_dir, "global_statistics.json"), stats)
    return stats


def save_best_model(best: dict, X: torch.Tensor, y_idx: torch.Tensor) -> str:
    best_dir = os.path.join(RESULTS_DIR, "best_model")
    if os.path.isdir(best_dir):
        shutil.rmtree(best_dir)
    name = dict(SUBEXPERIMENTS)[f"sub{best['subexperiment_id']}"]
    network = MLPNetwork(V=best["V"], W=best["W"], mu=best["mu"], std=best["std"])
    metadata = {
        "experiment_id": best["experiment_id"], "subexperiment_id": best["subexperiment_id"],
        "subexperiment": name, "seed": best["seed"],
        "config": config.to_dict(),
        "final": {"test_accuracy": best["metrics"]["accuracy"],
                  "final_training_loss": best["final_training_loss"],
                  "epochs_executed": best["epochs_executed"], "converged": best["converged"]},
        "metrics_test": best["metrics"],
        "confusion_matrix_test": best["confusion_matrix"].tolist(),
        "train_indices": best["train_idx"].tolist(), "test_indices": best["test_idx"].tolist(),
        "selection_rule": "máxima accuracy de prueba; desempate: menor loss final de entrenamiento; "
                          "después el primero en orden",
        "input": "X sin estandarizar (N×4); el modelo aplica mu/std de su fold de entrenamiento y agrega el bias",
        "feature_names": FEATURE_NAMES, "class_names": CLASS_NAMES,
    }
    save_model(best_dir, network, metadata, extra_tensors={"losses": torch.tensor(best["losses"])})
    write_epochs_csv(os.path.join(best_dir, "epochs.csv"), best["losses"])
    write_json(os.path.join(best_dir, "summary.json"), subexperiment_summary(best, y_idx))
    plot_subexperiment(best, f"Mejor · {subexperiment_title(best['experiment_id'], name)}", X, y_idx,
                       os.path.join(best_dir, "plots"))
    return best_dir

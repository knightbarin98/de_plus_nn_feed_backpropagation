from __future__ import annotations

import os

import torch

from config import CLASS_NAMES, DE_CONFIG, EPSILON, MAX_EPOCHS, RESULTS_DIR, SAVE_SUBEXPERIMENT_PLOTS, TRAINER
from network import MLPNetwork
from .logging_utils import write_confusion_csv, write_csv, write_epochs_csv, write_json
from .metrics import class_counts, classification_metrics
from .statistics import describe
from .visualization import plot_subexperiment

# sub1 entrena con A y prueba con B; sub2 al revés
SUBEXPERIMENTS = (("sub1", "A_to_B"), ("sub2", "B_to_A"))
N_CLASSES = len(CLASS_NAMES)


def subexperiment_title(experiment_id: int, name: str) -> str:
    return f"Exp {experiment_id:02d} · {name.replace('_to_', '→')}"


def stop_reason(sub: dict) -> str:
    if TRAINER == "de":
        return "max_generations"     # el DE siempre corre max_generations (should_stop no usa reached_optimum)
    return "converged" if sub["converged"] else "max_epochs"


def subexperiment_summary(sub: dict, y_idx: torch.Tensor) -> dict:
    losses = sub["losses"]
    return {
        "seed": sub["seed"],
        "train_class_counts": class_counts(y_idx[sub["train_idx"]], N_CLASSES),
        "test_class_counts": class_counts(y_idx[sub["test_idx"]], N_CLASSES),
        "standardizer": {"mu": sub["mu"], "std": sub["std"]},
        "training": {
            "trainer": TRAINER,
            "de_strategy": "DE/rand/1/bin" if TRAINER == "de" else None,
            "epochs_executed": sub["epochs_executed"],
            "max_epochs": DE_CONFIG.max_generations if TRAINER == "de" else MAX_EPOCHS,
            "epsilon": EPSILON,
            "converged": sub["converged"],
            "stop_reason": stop_reason(sub),
            "initial_training_loss": losses[0],
            "final_training_loss": sub["final_training_loss"],
            "cumulative_training_loss": sub["cumulative_training_loss"],
            "mean_training_loss": sub["mean_training_loss"],
            "V_final": sub["V"],
            "W_final": sub["W"],
        },
        "final": {"test_accuracy": sub["eval"]["accuracy"]},
        "confusion_matrix_test": sub["confusion_matrix"],
        "metrics_test": sub["metrics"],
        "train_loss_statistics": describe(losses),
    }


def save_subexperiment(sub: dict, title: str, X: torch.Tensor, y_idx: torch.Tensor, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    write_epochs_csv(os.path.join(out_dir, "epochs.csv"), sub["losses"])
    write_json(os.path.join(out_dir, "summary.json"), subexperiment_summary(sub, y_idx))

    torch.save({"V": sub["V"], "W": sub["W"], "mu": sub["mu"], "std": sub["std"]},
               os.path.join(out_dir, "weights.pt"))
    write_confusion_csv(os.path.join(out_dir, "confusion_matrix_test.csv"), sub["confusion_matrix"], CLASS_NAMES)

    # salidas de las 3 neuronas y activaciones de las neuronas ocultas de cada flor de prueba
    test_idx = sub["test_idx"]
    ev = sub["eval"]
    Z_test = MLPNetwork(V=sub["V"], W=sub["W"], mu=sub["mu"], std=sub["std"]).hidden(X[test_idx])   # (75×7)
    rows = [[int(test_idx[n]), int(ev["y_true"][n]), int(ev["y_pred"][n])]
            + ev["outputs"][n].tolist() + Z_test[n].tolist()
            for n in range(test_idx.numel())]
    write_csv(os.path.join(out_dir, "test_outputs.csv"),
              ["original_index", "true_class", "predicted_class", "output_0", "output_1", "output_2"]
              + [f"hidden_{j + 1}" for j in range(Z_test.shape[1])], rows)

    if SAVE_SUBEXPERIMENT_PLOTS:
        plot_subexperiment(sub, title, X, y_idx, os.path.join(out_dir, "plots"))


def experiment_aggregate(experiment: dict) -> dict:
    cm = experiment["confusion_matrix"]
    metrics = classification_metrics(cm)
    return {"accuracy": metrics["accuracy"], "metrics": metrics, "confusion_matrix": cm}


def experiment_summary(experiment: dict, aggregate: dict) -> dict:
    s1, s2 = experiment["sub1"], experiment["sub2"]
    acc = [s1["metrics"]["accuracy"], s2["metrics"]["accuracy"]]
    return {
        "experiment_id": experiment["experiment_id"],
        "experiment_seed": experiment["seed"],
        "test_accuracy_A_to_B": acc[0],
        "test_accuracy_B_to_A": acc[1],
        "test_accuracy_mean": sum(acc) / 2,
        "test_accuracy_aggregated": aggregate["accuracy"],
        "test_accuracy_statistics": describe(acc),
        "aggregated_confusion_matrix_test": aggregate["confusion_matrix"],
        "aggregated_metrics_test": aggregate["metrics"],
        "epochs": {"A_to_B": s1["epochs_executed"], "B_to_A": s2["epochs_executed"]},
        "converged": {"A_to_B": s1["converged"], "B_to_A": s2["converged"]},
    }


def save_experiment(experiment: dict, X: torch.Tensor, y_idx: torch.Tensor) -> dict:
    exp_id, seed = experiment["experiment_id"], experiment["seed"]
    exp_dir = os.path.join(RESULTS_DIR, f"experiment_{exp_id:02d}")
    os.makedirs(exp_dir, exist_ok=True)

    write_json(os.path.join(exp_dir, "experiment.json"), {
        "experiment_id": exp_id,
        "seed": seed,
        "A_indices": experiment["idx_A"],
        "B_indices": experiment["idx_B"],
        "A_class_counts": class_counts(y_idx[experiment["idx_A"]], N_CLASSES),
        "B_class_counts": class_counts(y_idx[experiment["idx_B"]], N_CLASSES),
    })
    in_A = set(experiment["idx_A"].tolist())
    write_csv(os.path.join(exp_dir, "split_indices.csv"), ["original_index", "set", "class"],
              [[i, "A" if i in in_A else "B", int(y_idx[i])] for i in range(X.shape[0])])

    aggregate = experiment_aggregate(experiment)
    write_json(os.path.join(exp_dir, "experiment_summary.json"), experiment_summary(experiment, aggregate))
    write_confusion_csv(os.path.join(exp_dir, "confusion_matrix_test_aggregated.csv"),
                        aggregate["confusion_matrix"], CLASS_NAMES)

    for key, name in SUBEXPERIMENTS:
        save_subexperiment(experiment[key], subexperiment_title(exp_id, name), X, y_idx,
                           os.path.join(exp_dir, f"subexperiment_{name}"))
    return aggregate


def sub_summary_row(experiment_id: int, subexperiment_id: int, sub: dict, train_set: str, test_set: str) -> dict:
    m = sub["metrics"]
    row = {
        "experiment_id": experiment_id, "subexperiment_id": subexperiment_id,
        "subexperiment": f"{train_set}_to_{test_set}", "seed": sub["seed"],
        "epochs_executed": sub["epochs_executed"], "converged": sub["converged"],
        "final_training_loss": sub["final_training_loss"],
        "cumulative_training_loss": sub["cumulative_training_loss"],
        "mean_training_loss": sub["mean_training_loss"],
        "test_accuracy": m["accuracy"], "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
    }
    for k, c in enumerate(CLASS_NAMES):
        row[f"precision_{c}"] = m["precision_per_class"][k]
    return row


def exp_summary_row(experiment: dict, aggregate: dict) -> dict:
    acc_ab = experiment["sub1"]["metrics"]["accuracy"]
    acc_ba = experiment["sub2"]["metrics"]["accuracy"]
    return {
        "experiment_id": experiment["experiment_id"], "seed": experiment["seed"],
        "test_accuracy_A_to_B": acc_ab, "test_accuracy_B_to_A": acc_ba,
        "test_accuracy_mean": (acc_ab + acc_ba) / 2, "test_accuracy_aggregated": aggregate["accuracy"],
    }

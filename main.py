from __future__ import annotations

import os
import logging
import torch

from config import DEVICE, DATA_CSV, NO_EXPERIMENTS as NUM_EXPERIMENTS, BASE_SEED, TEST_SIZE, RESULTS_DIR
from data import load_iris_raw, build_tensors
from experiment import run_experiment
from utils.results import save_experiment, sub_summary_row, exp_summary_row
from utils.global_results import save_global, save_best_model

def setup_logger() -> logging.Logger:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = logging.getLogger("iris")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"), mode="w")
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def better(a: dict, b: dict | None) -> bool:
    """True si el subexperimento `a` es mejor que `b` (acc alta, luego loss baja)."""
    if b is None:
        return True
    ka = (a["metrics"]["accuracy"], -a["final_training_loss"])
    kb = (b["metrics"]["accuracy"], -b["final_training_loss"])
    return ka > kb

def run_all_experiments(verbose: bool = True) -> torch.Tensor:
    log = setup_logger()
    feats, labels = load_iris_raw(DATA_CSV)
    X, y_idx, Y = build_tensors(feats, labels, DEVICE)
    assert X.shape == (150,4) and Y.shape == (150, 3)
    X_cpu, y_idx_cpu = X.cpu(), y_idx.cpu()

    log.info("Iris cargado: %s muestras, device=%s", X.shape[0], DEVICE)

    confussion_matrix_global = torch.zeros(3,3,dtype=torch.long)
    finals, cumulatives, all_epoch_losses = [], [], []
    sub_rows, exp_rows = [], []
    epochs_list, accuracies = [], []
    best = None

    for e in range(1, NUM_EXPERIMENTS + 1):
        experiment = run_experiment(X, y_idx, Y, e, BASE_SEED + e, DEVICE)
        confussion_matrix_global += experiment["confusion_matrix"]

        for sid, (sub, tr, te) in enumerate([(experiment["sub1"], "A", "B"), (experiment["sub2"], "B", "A")], start=1):
            finals.append(sub["final_training_loss"])
            cumulatives.append(sub["cumulative_training_loss"])
            all_epoch_losses.extend(sub["losses"])
            epochs_list.append(sub["epochs_executed"])
            sub_rows.append(sub_summary_row(e, sid, sub, tr, te))
            assert sub["eval"]["y_true"].shape[0] == TEST_SIZE

            # log de estadísticos + métricas por subexperimento
            m = sub["metrics"]
            log.info("exp %02d sub %d | epochs=%d conv=%s | final_loss=%.6f cum=%.2f "
                     "mean=%.6f | acc=%.4f macroP=%.4f | P[set,ver,vir]=[%.3f,%.3f,%.3f]",
                     e, sid, sub["epochs_executed"], sub["converged"],
                     sub["final_training_loss"], sub["cumulative_training_loss"],
                     sub["mean_training_loss"], m["accuracy"], m["macro_precision"],
                     *m["precision_per_class"])

            # seguimiento del mejor modelo
            if better(sub, best):
                best = dict(sub)
                best["experiment_id"] = e
                best["subexperiment_id"] = sid

        aggregate = save_experiment(experiment, X_cpu, y_idx_cpu)
        exp_rows.append(exp_summary_row(experiment, aggregate))
        accuracies.append(aggregate["accuracy"])
        if verbose:
            log.info("== experimento %02d | accuracy OOF=%.4f", e, aggregate["accuracy"])

    # validaciones globales
    assert NUM_EXPERIMENTS == 35 and len(sub_rows) == 70
    assert confussion_matrix_global.sum().item() == 5250, confussion_matrix_global.sum().item()

    save_global(confussion_matrix_global, finals, cumulatives, all_epoch_losses, sub_rows, exp_rows)
    best_dir = save_best_model(best, X_cpu, y_idx_cpu)

    log.info("===== RESUMEN GLOBAL =====")
    log.info("CM global (filas=real, cols=pred):\n%s", confussion_matrix_global)
    log.info("total predicciones=%d | accuracy global=%.4f",
             confussion_matrix_global.sum().item(), confussion_matrix_global.diag().sum().item() / 5250)
    log.info("accuracy media por experimento=%.4f", sum(accuracies) / len(accuracies))
    log.info("épocas: min=%d max=%d media=%.0f",
             min(epochs_list), max(epochs_list), sum(epochs_list) / len(epochs_list))
    log.info("MEJOR modelo: exp %02d sub %d | acc=%.4f macroP=%.4f -> %s",
             best["experiment_id"], best["subexperiment_id"],
             best["metrics"]["accuracy"], best["metrics"]["macro_precision"], best_dir)
    return confussion_matrix_global




if __name__ == "__main__":
    run_all_experiments()

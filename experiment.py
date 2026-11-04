from __future__ import annotations

import torch

from config import (MAX_EPOCHS, EPSILON, LEARNING_RATE, HIDDEN_NEURONS, CLASS_NAMES, TRAINER)
from network import (init_weights, forward, mse_loss, backward, update_weights)
from data import fit_standardizer, apply_standarizer, add_bias
from de_training import train_subexperiment_de
from utils.metrics import classification_metrics

def train_subexperiment(X_aug_train: torch.Tensor, Y_train: torch.Tensor, seed: int, device: torch.device,
                        alpha: float = LEARNING_RATE, max_epochs: int = MAX_EPOCHS,
                        epsilon: float = EPSILON) -> tuple[torch.Tensor, torch.Tensor, list[float], bool]:
    """
        Entrena UNA red desde cero (reinicializa pesos con `seed`).
        Devuelve V, W, errores (lista float por época), convergencia (bool).
    """
    V, W = init_weights(seed, device)
    losses = []
    converged = False
    for epoch in range(1, max_epochs + 1):
        Y_hat, cache = forward(X_aug_train,V, W)
        E = mse_loss(Y_hat, Y_train).item()
        losses.append(E)

        #criterio epsilon 
        if E <= epsilon:
            converged = True
            break
        dE_dV, dE_dW = backward(X_aug_train, Y_train, W, cache)
        update_weights(V, W, dE_dV, dE_dW)

    return V, W, losses, converged

def evaluate(X_aug_test: torch.Tensor, y_idx_test: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> dict:
    """
    Feedforward sobre test, argmax -> clase. Devuelve dict con predicciones.
    """
    Y_hat, _ = forward(X_aug_test, V, W)     # 75×3
    y_pred = Y_hat.argmax(dim=1)               # 75
    accuracy = (y_pred == y_idx_test).float().mean().item()
    return {
        "y_true": y_idx_test.cpu(),
        "y_pred": y_pred.cpu(),
        "outputs": Y_hat.detach().cpu(),       # 75×3, para guardar output_0..2
        "accuracy": accuracy,
    }

def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        confusion_matrix[t, p] += 1
    return confusion_matrix

def create_two_folds(y_idx: torch.Tensor, seed: int, device: torch.device, per_class: int = 25,
                     num_classes: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    y_cpu = y_idx.cpu()
    part_A, part_B = [], []
    for c in range(num_classes):
        cls_idx = (y_cpu == c).nonzero(as_tuple=True)[0] #indices de la clase c
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=gen)] #number of elements
        part_A.append(perm[:per_class])
        part_B.append(perm[per_class: 2*per_class])
    idx_A = torch.cat(part_A).to(device)
    idx_B = torch.cat(part_B).to(device)
    return idx_A, idx_B

def check_folds(idx_A: torch.Tensor, idx_B: torch.Tensor, y_idx: torch.Tensor, N: int = 150,
                per_class: int = 25, num_classes: int = 3) -> None:
    a, b = set(idx_A.tolist()), set(idx_B.tolist())
    assert len(a) == N // 2 and len(b) == N // 2, (len(a), len(b))
    assert a.isdisjoint(b), "A ∩ B != vacío"
    assert a | b == set(range(N)), "A ∪ B != todas"
    # estratificación: 25 de cada clase en cada fold
    y_cpu = y_idx.cpu()
    for name, idx in (("A", idx_A), ("B", idx_B)):
        counts = torch.bincount(y_cpu[idx.cpu()], minlength=num_classes)
        assert torch.all(counts == per_class), f"fold {name} no estratificado: {counts.tolist()}"

def run_subexperiment(X: torch.Tensor, y_idx: torch.Tensor, Y: torch.Tensor, train_idx: torch.Tensor,
                      test_idx: torch.Tensor, seed: int, device: torch.device) -> dict:
    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr = Y[train_idx]
    yi_te = y_idx[test_idx]

    Xtr_aug = add_bias(X_tr)
    Xte_aug = add_bias(X_te)

    if TRAINER == "de":
        V, W, losses, converged = train_subexperiment_de(Xtr_aug, Y_tr, seed, device)
    else:
        V, W, losses, converged = train_subexperiment(Xtr_aug, Y_tr, seed, device)

    ev = evaluate(Xte_aug, yi_te, V, W)
    confussion_matrix = compute_confusion_matrix(ev["y_true"], ev["y_pred"])
    metrics = classification_metrics(confussion_matrix)

    return {
        "seed": seed,
        "losses": losses,
        "converged": converged,
        "epochs_executed": len(losses),
        "final_training_loss": losses[-1],
        "cumulative_training_loss": float(sum(losses)),
        "mean_training_loss": float(sum(losses) / len(losses)),
        "eval": ev,
        "confusion_matrix": confussion_matrix,
        "metrics": metrics,
        "mu": mu.cpu(), "std": std.cpu(),
        "train_idx": train_idx.cpu(), "test_idx": test_idx.cpu(),
        "V": V.detach().cpu(), "W": W.detach().cpu(),   # mejor-modelo / reproducción
    }

def run_experiment(X: torch.Tensor, y_idx: torch.Tensor, Y: torch.Tensor, experiment_id: int, base_seed: int,
                   device: torch.device) -> dict:
    idx_A, idx_B = create_two_folds(y_idx, base_seed, device)
    check_folds(idx_A, idx_B, y_idx, X.shape[0])

    subexperiment_1 = run_subexperiment(X, y_idx, Y, idx_A, idx_B, base_seed, device)
    subexperiment_2 = run_subexperiment(X, y_idx, Y, idx_B, idx_A, base_seed, device)

    confussion_matrix_experimento = subexperiment_1["confusion_matrix"] + subexperiment_2["confusion_matrix"]
    assert confussion_matrix_experimento.sum().item() == 150, "predicciones OOF != 150"

    return {"experiment_id": experiment_id, "seed": base_seed,
            "idx_A": idx_A.cpu(), "idx_B": idx_B.cpu(),
            "sub1": subexperiment_1, "sub2": subexperiment_2, "confusion_matrix": confussion_matrix_experimento}



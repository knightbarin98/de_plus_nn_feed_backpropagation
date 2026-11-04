from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo archivos PNG
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import CLASS_NAMES, EPSILON, HIDDEN_NEURONS, ITERATION_START, TRAINER  # noqa: E402
from network import MLPNetwork  # noqa: E402

# Paleta categórica (primeros 3 espacios validados para daltonismo entre sí)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # azul, naranja, aqua
TRAIN_COLOR, TEST_COLOR = SERIES[0], SERIES[1]
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
CLASS_BAND = ["#eef4fc", "#fdf0ea", "#eaf7f1"]       # fondo muy claro por clase verdadera
N_CLASSES = 3
ITERATION_LABEL = "Época" if TRAINER == "backprop" else "Generación"
ITERATIONS_LABEL = "Épocas" if TRAINER == "backprop" else "Generaciones"
TRAINER_LABEL = "backprop" if TRAINER == "backprop" else "DE"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.6, "figure.dpi": 110,
})


def display_names(class_names: list) -> list:
    return [c.replace("Iris-", "") for c in class_names]    # "Iris-setosa" -> "setosa" en las gráficas


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _np(t):
    return t.detach().cpu().double().numpy() if isinstance(t, torch.Tensor) else t


def plot_error_curve(epochs, errors, title: str, color: str, path: str, epsilon: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(_np(epochs), _np(errors), color=color)
    if epsilon is not None:
        ax.axhline(epsilon, color=MUTED, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], epsilon, f" ε = {epsilon:g}", va="bottom", ha="right", color=TEXT_2, fontsize=8)
    ax.set_xlabel(ITERATION_LABEL)
    ax.set_ylabel("E = 1/(2N) Σ (ŷ − y)²  (log)")
    ax.set_title(title)
    _save(fig, path)


def plot_confusion_matrix(cm, class_names: list, title: str, path: str) -> None:
    cm = _np(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=max(cm.max(), 1))
    ax.grid(False)
    n = len(class_names)
    for i in range(n):
        row_total = cm[i].sum()
        for j in range(n):
            v = cm[i, j]
            pct = f"\n{100 * v / row_total:.0f}%" if row_total else ""
            ax.text(j, i, f"{int(v)}{pct}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.55 * cm.max() else TEXT)
    ax.set_xticks(range(n), class_names)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def _class_bands(ax, y_true, class_names: list) -> None:
    y = list(_np(y_true).astype(int))
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            ax.axvspan(start - 0.5, i - 0.5, color=CLASS_BAND[y[start]], zorder=0, linewidth=0)
            ax.text((start + i - 1) / 2, 1.0, class_names[y[start]], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXT_2)
            start = i


def plot_neuron_responses(values, y_true, y_pred, class_names: list, title: str, ylabel: str,
                          label_prefix: str, path: str, target_lines: bool = True) -> None:
    v = _np(values)
    N = v.shape[0]
    x = list(range(N))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _class_bands(ax, y_true, class_names)
    for k in range(N_CLASSES):
        ax.plot(x, v[:, k], color=SERIES[k], marker="o", markersize=3, linewidth=1.2,
                label=f"{label_prefix}_{k}  ({class_names[k]})")
    if y_pred is not None:
        yt, yp = _np(y_true).astype(int), _np(y_pred).astype(int)
        wrong = [n for n in range(N) if yt[n] != yp[n]]
        if wrong:
            ax.scatter(wrong, [v[n, yp[n]] for n in wrong], marker="x", s=55, color=TEXT, zorder=5,
                       label=f"mal clasificada ({len(wrong)})")
    if target_lines:
        for t in (0.0, 1.0):
            ax.axhline(t, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xlabel("Muestra del conjunto de prueba (en el orden del fold: agrupadas por clase)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, path)


def plot_hidden_activations(Z, y_true, class_names: list, title: str, path: str) -> None:
    Zn = _np(Z)                                       # (N×7)
    N, H = Zn.shape
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(Zn.T, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.grid(False)
    y = list(_np(y_true).astype(int))
    start = 0
    for i in range(1, N + 1):
        if i == N or y[i] != y[start]:
            if i < N:
                ax.axvline(i - 0.5, color=SURFACE, linewidth=2)
            ax.text((start + i - 1) / 2, 1.0, class_names[y[start]], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXT_2)
            start = i
    ax.set_yticks(range(H), [f"z_{j + 1}" for j in range(H)])
    ax.set_xlabel("Muestra del conjunto de prueba (en el orden del fold: agrupadas por clase)")
    ax.set_title(title, pad=16)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="σ(z_in)")
    _save(fig, path)


def plot_accuracy_distribution(values, title: str, path: str) -> None:
    v = [round(float(a), 6) for a in _np(values)]
    uniq = sorted(set(v))
    counts = [v.count(u) for u in uniq]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar([100 * u for u in uniq], counts, width=0.9, color=SERIES[0])
    ax.set_xlabel("Accuracy de prueba (%)")
    ax.set_ylabel("Número de subexperimentos")
    ax.set_title(title)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _save(fig, path)


def plot_experiment_accuracies(acc_ab, acc_ba, acc_mean, title: str, path: str) -> None:
    ids = list(range(1, len(acc_ab) + 1))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.vlines(ids, [min(a, b) * 100 for a, b in zip(acc_ab, acc_ba)],
              [max(a, b) * 100 for a, b in zip(acc_ab, acc_ba)], color=GRID, linewidth=3, zorder=1)
    ax.scatter(ids, [a * 100 for a in acc_ab], s=26, color=SERIES[0], label="A → B", zorder=3)
    ax.scatter(ids, [a * 100 for a in acc_ba], s=26, color=SERIES[1], label="B → A", zorder=3)
    ax.scatter(ids, [a * 100 for a in acc_mean], s=46, marker="_", color=TEXT, linewidth=2,
               label="promedio del experimento", zorder=4)
    ax.set_xticks(ids)
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Accuracy de prueba (%)")
    ax.set_title(title, pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)   # fuera del eje: no tapa los puntos
    _save(fig, path)


def plot_all_error_curves(curves, title: str, ylabel: str, color: str, path: str,
                          epsilon: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for c in curves:
        ax.semilogy(range(ITERATION_START, ITERATION_START + len(c)), _np(c), color=color, alpha=0.25, linewidth=0.9)
    if epsilon is not None:
        ax.axhline(epsilon, color=MUTED, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], epsilon, f" ε = {epsilon:g}", va="bottom", ha="right", color=TEXT_2, fontsize=8)
    ax.set_xlabel(ITERATION_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def plot_histogram(values, title: str, xlabel: str, path: str, color: str | None = None) -> None:
    v = [float(a) for a in _np(values)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(v, bins=min(20, max(5, len(set(v)))), color=color or SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    _save(fig, path)


def plot_subexperiment(sub: dict, title: str, X: torch.Tensor, y_idx: torch.Tensor, plot_dir: str) -> None:
    names = display_names(CLASS_NAMES)
    network = MLPNetwork(V=sub["V"], W=sub["W"])
    X_test = X[sub["test_idx"]]                       # (75×4) en el orden del fold
    losses = sub["losses"]
    ev = sub["eval"]

    plot_error_curve(range(ITERATION_START, ITERATION_START + len(losses)), losses,
                     f"{title} · error de entrenamiento ({TRAINER_LABEL})",
                     TRAIN_COLOR, os.path.join(plot_dir, "01_train_loss.png"), epsilon=EPSILON)
    plot_confusion_matrix(sub["confusion_matrix"], names, f"{title} · prueba",
                          os.path.join(plot_dir, "02_confusion_test.png"))
    plot_neuron_responses(ev["outputs"], ev["y_true"], ev["y_pred"], names,
                          f"{title} · respuesta de las 3 neuronas de salida (prueba)", "Salida ŷ_k", "output",
                          os.path.join(plot_dir, "03_output_neurons_test.png"))
    plot_hidden_activations(network.hidden(X_test), ev["y_true"], names,
                            f"{title} · activaciones de las {HIDDEN_NEURONS} neuronas ocultas (prueba)",
                            os.path.join(plot_dir, "04_hidden_activations_test.png"))


def plot_global(sub_rows: list, exp_rows: list, cm_global: torch.Tensor, curves: list, plot_dir: str) -> None:
    n = len(sub_rows)
    names = display_names(CLASS_NAMES)
    plot_accuracy_distribution([r["test_accuracy"] for r in sub_rows],
                               f"Accuracy de prueba de los {n} subexperimentos",
                               os.path.join(plot_dir, "test_accuracy_subexperiments.png"))
    plot_experiment_accuracies([e["test_accuracy_A_to_B"] for e in exp_rows],
                               [e["test_accuracy_B_to_A"] for e in exp_rows],
                               [e["test_accuracy_mean"] for e in exp_rows],
                               "Accuracy de prueba por experimento", os.path.join(plot_dir, "experiments_accuracy.png"))
    plot_confusion_matrix(cm_global, names, f"Prueba · suma de los {n} subexperimentos",
                          os.path.join(plot_dir, "confusion_test_global.png"))
    plot_all_error_curves(curves, f"Error de entrenamiento ({TRAINER_LABEL}) · {n} subexperimentos",
                          "E entrenamiento (log)",
                          TRAIN_COLOR, os.path.join(plot_dir, "train_loss_curves.png"), epsilon=EPSILON)
    plot_histogram([r["epochs_executed"] for r in sub_rows], f"{ITERATIONS_LABEL} registradas", ITERATIONS_LABEL,
                   os.path.join(plot_dir, "epochs_hist.png"))
    plot_histogram([r["final_training_loss"] for r in sub_rows], "Error final de entrenamiento",
                   "E final", os.path.join(plot_dir, "final_loss_hist.png"), color=SERIES[1])

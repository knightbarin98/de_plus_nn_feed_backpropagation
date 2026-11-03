from __future__ import annotations

from dataclasses import dataclass

import torch

from config import (INPUT_WITH_BIAS, HIDDEN_NEURONS, HIDDEN_WITH_BIAS, OUTPUT_NEURONS, LEARNING_RATE)
from data import add_bias, apply_standarizer

def init_weights(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Inicializacion aleatoria uniforme"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lim1 = 1.0 / (INPUT_WITH_BIAS ** 0.5)
    lim2 = 1.0/ (HIDDEN_WITH_BIAS ** 0.5)
    U1 = torch.rand(INPUT_WITH_BIAS, HIDDEN_NEURONS, generator=gen)
    U2 = torch.rand(HIDDEN_WITH_BIAS, OUTPUT_NEURONS, generator=gen)
    V = ((U1 * 2.0 - 1.0) * lim1).to(device) # U[-lim1, lim1]
    W = ((U2 * 2.0 - 1.0) * lim2).to(device) # U[-lim2, lim2]
    return V, W

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoide estable, elemento a elemento. Sin torch.sigmoid."""
    pos = x >= 0
    z = torch.empty_like(x)
    # rama x >= 0
    z[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    # rama x < 0
    ex = torch.exp(x[~pos])
    z[~pos] = ex / (1.0 + ex)
    return z

def sigmoid_derivative_sigmoid_notcalculated(z: torch.Tensor) -> torch.Tensor:
    """σ'(z) = σ(z)(1-σ(z)). Recibe la pre-activación z."""
    s = sigmoid(z)
    return s * (1.0 - s)

def sigmoid_derivative_sigmoid_calculated(a: torch.Tensor) -> torch.Tensor:
    """σ'(z) = a(1-a) cuando ya se tiene a = σ(z). Ya habia sido calculada en el feedforward"""
    return a * (1.0 - a)

def forward(X_aug: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> tuple[torch.Tensor, dict]:
    Zin = X_aug @ V # Nx5 @ 5x7 = Nx7
    Z = sigmoid(Zin)
    Z_aug = add_bias(Z)
    Yin = Z_aug @ W # Nx7 @ 7x3 = Nx3
    Y_hat = sigmoid(Yin)
    cache = {"Zin":Zin, "Z":Z, "Z_aug":Z_aug, "Yin":Yin, "Y_hat":Y_hat}
    return Y_hat, cache

def mse_loss(Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Escalar: (1/(2N)) * sum((Y_hat - Y)^2). Este es EL 'loss'."""
    N = Y.shape[0]
    diff = Y_hat - Y
    return (diff * diff).sum() / (2.0 * N)

def mse_grad_output(Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """dL/dY_hat = (1/N)(Y_hat - Y)  -> N×3."""
    N = Y.shape[0]
    return (Y_hat - Y) / N

def backward(X_aug: torch.Tensor, Y: torch.Tensor, W: torch.Tensor,
             cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backpropagation manual. Devuelve dW1 (5×7) y dW2 (8×3).
    No usa autograd.
    """
    Y_hat = cache["Y_hat"] # N×3
    Z = cache["Z"]  # N×7
    Z_aug = cache["Z_aug"] # N×8

    # ---- capa salida a capa oculta ----
    dE_dY_hat = mse_grad_output(Y_hat, Y) # Nx3 = (1/N) (^y - y)
    delta_k = dE_dY_hat * sigmoid_derivative_sigmoid_calculated(Y_hat) # N×3  ⊙ σ'(Z2)
    dE_dW = Z_aug.T @ delta_k # 8×3   (12)

    delta_in_j_aug = delta_k @ W.T # Nx8
    delta_in_j = delta_in_j_aug[:, :HIDDEN_NEURONS]  # Nx7 (quitar bias)

    delta_j_prima = delta_in_j * sigmoid_derivative_sigmoid_calculated(Z) # Nx7
    dE_dV = X_aug.T @ delta_j_prima # 5xN @ Nx7 = 5x7
    return dE_dV, dE_dW

def update_weights(V: torch.Tensor, W: torch.Tensor, dE_dV: torch.Tensor, dE_dW: torch.Tensor,
                   alpha: float = LEARNING_RATE) -> tuple[torch.Tensor, torch.Tensor]:
    V -= alpha * dE_dV
    W -= alpha * dE_dW
    return V, W


@dataclass
class MLPNetwork:
    V: torch.Tensor          # (5, 7)  entrada+bias -> oculta
    W: torch.Tensor          # (8, 3)  oculta+bias -> salida
    mu: torch.Tensor         # (4,)    media del fold de entrenamiento
    std: torch.Tensor        # (4,)    desviación poblacional del fold de entrenamiento

    def prepare(self, X: torch.Tensor) -> torch.Tensor:
        # (N×4) estandarizada con mu/std de entrenamiento -> + columna de unos -> (N×5)   ej. (75×4) -> (75×5)
        return add_bias(apply_standarizer(X, self.mu, self.std))

    def hidden(self, X: torch.Tensor) -> torch.Tensor:
        _, cache = forward(self.prepare(X), self.V, self.W)
        return cache["Z"]                                # (N×7)   ej. (75×7)

    def outputs(self, X: torch.Tensor) -> torch.Tensor:
        Y_hat, _ = forward(self.prepare(X), self.V, self.W)
        return Y_hat                                     # (N×5) @ (5×7) -> σ, + bias (N×8) @ (8×3) -> σ (N×3)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.outputs(X), dim=1)      # (N×3) -> índice del máximo por fila -> (N,)

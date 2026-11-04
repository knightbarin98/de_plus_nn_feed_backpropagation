from __future__ import annotations

import torch

from algorithm import DifferentialEvolution, OptimizationResult
from config import (DE_CONFIG, DE_LOWER_BOUND, DE_UPPER_BOUND, EPSILON, HIDDEN_NEURONS, HIDDEN_WITH_BIAS,
                    INPUT_WITH_BIAS, OUTPUT_NEURONS, DEConfig)
from network import forward, mse_loss, sigmoid

N_V = INPUT_WITH_BIAS * HIDDEN_NEURONS        # 5·7 = 35 primeras componentes del genoma


def decode_genome(genome: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # [fig. 1] genoma = [V aplanada fila por fila | W aplanada fila por fila]
    # (d,) -> V (5×7), W (8×3)      ó      (P×d) -> V (P×5×7), W (P×8×3)
    batch = genome.shape[:-1]
    V = genome[..., :N_V].reshape(*batch, INPUT_WITH_BIAS, HIDDEN_NEURONS)
    W = genome[..., N_V:].reshape(*batch, HIDDEN_WITH_BIAS, OUTPUT_NEURONS)
    return V, W


def encode_weights(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # V (5×7), W (8×3) -> (35,) | (24,) -> (59,)
    return torch.cat([V.reshape(-1), W.reshape(-1)])


def forward_population(X_aug: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # el mismo feedforward de network.forward, para las P redes de la población a la vez
    Zin = X_aug @ V                    # (N×5) @ (P×5×7) -> (P×N×7)   ej. (75×5) @ (20×5×7) -> (20×75×7)
    Z = sigmoid(Zin)
    ones = torch.ones((*Z.shape[:-1], 1), dtype=Z.dtype, device=Z.device)    # (P×N×1)
    Z_aug = torch.cat([Z, ones], dim=-1)                                     # (P×N×8): bias a la derecha
    Yin = Z_aug @ W                    # (P×N×8) @ (P×8×3) -> (P×N×3)
    return sigmoid(Yin)


def network_error_population(X_aug: torch.Tensor, Y: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
    """Función objetivo del DE: el mismo error de network.mse_loss, (1/(2N)) Σ (ŷ - y)², por individuo."""
    V, W = decode_genome(population)
    Y_hat = forward_population(X_aug, V, W)        # (P×N×3)
    N = Y.shape[0]
    diff = Y_hat - Y                               # (P×N×3) - (N×3) -> (P×N×3)
    return (diff * diff).sum(dim=(1, 2)) / (2.0 * N)    # suma por red -> (P,)


def test_loss_history(X_aug_test: torch.Tensor, Y_test: torch.Tensor, best_position_history: list[torch.Tensor],
                      chunk_size: int = 1000) -> list[float]:
    # error de prueba del mejor genoma de cada generación, calculado al terminar: el DE nunca ve el conjunto de prueba
    genomes = torch.stack(best_position_history)      # (G+1×d) en CPU
    errors = [network_error_population(X_aug_test, Y_test, chunk.to(X_aug_test.device))
              for chunk in torch.split(genomes, chunk_size)]
    return torch.cat(errors).cpu().tolist()


def train_subexperiment_de(X_aug_train: torch.Tensor, Y_train: torch.Tensor, seed: int, device: torch.device,
                           de_config: DEConfig = DE_CONFIG,
                           epsilon: float = EPSILON) -> tuple[torch.Tensor, torch.Tensor, list[float], bool,
                                                              OptimizationResult]:
    """
        Entrena UNA red con Evolución Diferencial (DE/rand/1/bin), desde una población nueva (`seed`).
        Devuelve lo mismo que train_subexperiment: V, W, errores (mejor fitness por generación), convergencia;
        y además el OptimizationResult del DE.
    """
    rng = torch.Generator(device=device).manual_seed(seed)

    def objective(population: torch.Tensor) -> torch.Tensor:
        return network_error_population(X_aug_train, Y_train, population)

    de = DifferentialEvolution(objective, DE_LOWER_BOUND, DE_UPPER_BOUND, de_config, rng,
                               target_fitness=0.0, dtype=X_aug_train.dtype)
    result = de.run()
    V, W = decode_genome(result.best_position)
    losses = result.fitness_history                # generación 0 (población inicial) ... max_generations

    # el fitness del mejor genoma debe ser exactamente el error de la red con network.forward / mse_loss
    E = mse_loss(forward(X_aug_train, V, W)[0], Y_train).item()
    assert abs(E - result.best_fitness) <= 1e-5 * max(1.0, E), (E, result.best_fitness)

    converged = result.best_fitness <= epsilon
    return V, W, losses, converged, result

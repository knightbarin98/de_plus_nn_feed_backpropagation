from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

NO_EXPERIMENTS: int = 35
K_FOLDS: int = 2
TRAIN_SIZE: int = 75
TEST_SIZE: int = 75
TOTAL_SAMPLE: int = 150

MAX_EPOCHS: int = 10000
EPSILON: float = 0.0001
LEARNING_RATE: float = 0.5

INPUT_FEATURE: int = 4
INPUT_WITH_BIAS: int = 5
HIDDEN_NEURONS: int = 7
HIDDEN_WITH_BIAS: int = 8
OUTPUT_NEURONS: int = 3

MODE_DECIMALS: int = 6
STD_POPULATION: bool = True

BASE_SEED: int = 42

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# método de entrenamiento: "de" (Evolución Diferencial, como en el artículo) | "backprop"
TRAINER: str = "de"

# genoma = todos los pesos de la red (fig. 1 del artículo): V (5×7) aplanada y después W (8×3) aplanada
GENOME_DIMENSION: int = INPUT_WITH_BIAS * HIDDEN_NEURONS + HIDDEN_WITH_BIAS * OUTPUT_NEURONS   # 35 + 24 = 59
DE_LOWER_BOUND: float = -10.0
DE_UPPER_BOUND: float = 10.0


# hiperparámetros de ../de-algorithm/config.py (DE/rand/1/bin), salvo población y dimensión
@dataclass(frozen=True)
class DEConfig:
    population_size: int = 20                   # artículo: 20 individuos
    dimension: int = GENOME_DIMENSION           # 59 pesos de la red
    F: float = 0.9
    CR: float = 0.8
    max_generations: int = 20000
    tolerance: float = 1e-8


DE_CONFIG: DEConfig = DEConfig()

DATA_CSV: str = "iris.csv"
RESULTS_DIR: str = "results" if TRAINER == "backprop" else "results_de"
# nombre y numeración de cada fila del historial de entrenamiento
ITERATION_NAME: str = "epoch" if TRAINER == "backprop" else "generation"
ITERATION_START: int = 1 if TRAINER == "backprop" else 0      # DE guarda también la generación 0 (población inicial)
SAVE_SUBEXPERIMENT_PLOTS: bool = True

CLASS_NAMES: list[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
FEATURE_NAMES: list[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def to_dict() -> dict:
    # todas las constantes en MAYÚSCULAS de este módulo (para guardarlas junto a los resultados)
    def plain(v):
        if isinstance(v, torch.device):
            return str(v)
        return asdict(v) if isinstance(v, DEConfig) else v
    return {k: plain(v) for k, v in globals().items() if k.isupper()}

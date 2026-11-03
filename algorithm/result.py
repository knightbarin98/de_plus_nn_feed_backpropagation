from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OptimizationResult:
    best_position: torch.Tensor
    best_fitness: float
    evaluations: int
    generations: int

    fitness_history: list[float] = field(default_factory=list)
    evaluations_history: list[int] = field(default_factory=list)
    best_position_history: list[torch.Tensor] = field(default_factory=list)

    population_history: list[torch.Tensor] = field(default_factory=list)

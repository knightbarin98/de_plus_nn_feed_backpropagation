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

    def __str__(self) -> str:
        # los historiales tienen una entrada por generación: solo se imprime su tamaño, no su contenido
        best_position = ", ".join(f"{v:.6f}" for v in self.best_position.tolist())
        return (f"best_fitness={self.best_fitness:.10g} evaluations={self.evaluations} "
                f"generations={self.generations} fitness_history={len(self.fitness_history)} "
                f"evaluations_history={len(self.evaluations_history)} "
                f"best_position_history={len(self.best_position_history)} "
                f"population_history={len(self.population_history)} best_position=[{best_position}]")

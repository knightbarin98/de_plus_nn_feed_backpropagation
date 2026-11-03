from __future__ import annotations

from collections.abc import Callable

import torch

from algorithm.result import OptimizationResult
from config import DEConfig

# recibe la población completa (P×d) y devuelve el fitness de cada individuo (P,)
ObjectiveFunction = Callable[[torch.Tensor], torch.Tensor]


class DifferentialEvolution:
    def __init__(
        self,
        objective: ObjectiveFunction,
        lower_bound: float | torch.Tensor,
        upper_bound: float | torch.Tensor,
        config: DEConfig,
        rng: torch.Generator,
        track_population: bool = False,
        target_fitness: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if config.population_size < 4:
            raise ValueError("DE/rand/1 needs population_size >= 4 (x_i, x_r1, x_r2, x_r3 distinct).")

        self.objective = objective
        self.config = config
        self.rng = rng
        self.track_population = track_population
        self.target_fitness = target_fitness

        # todos los tensores viven en el dispositivo del generador (cuda si está disponible)
        self.device = rng.device
        self.dtype = dtype
        self.population_size = config.population_size
        self.dimension = config.dimension
        self.lower_bound = torch.broadcast_to(
            torch.as_tensor(lower_bound, dtype=dtype, device=self.device), (self.dimension,))   # (d,)
        self.upper_bound = torch.broadcast_to(
            torch.as_tensor(upper_bound, dtype=dtype, device=self.device), (self.dimension,))   # (d,)

        self.population = torch.empty((self.population_size, self.dimension), dtype=dtype, device=self.device)
        self.fitness = torch.full((self.population_size,), float("inf"), dtype=dtype, device=self.device)
        self.evaluations = 0
        self.generation = 0
        self.best_position = torch.empty(self.dimension, dtype=dtype, device=self.device)
        # escalar 0-d en el dispositivo: compararlo no obliga a sincronizar con la GPU en cada generación
        self.best_fitness = torch.tensor(float("inf"), dtype=dtype, device=self.device)

    def initialize_population(self) -> torch.Tensor:
        span = self.upper_bound - self.lower_bound                                        # (d,)
        U = torch.rand((self.population_size, self.dimension), generator=self.rng,
                       dtype=self.dtype, device=self.device)                              # (P×d) en [0, 1)
        return self.lower_bound + U * span                                                # (d,) + (P×d)*(d,) -> (P×d)

    def evaluate(self, x_i: torch.Tensor) -> torch.Tensor:
        self.evaluations += 1
        return self.objective(x_i[None, :])[0]          # (d,) -> (1×d) -> (1,) -> escalar

    def evaluate_population(self, population: torch.Tensor) -> torch.Tensor:
        # un solo llamado para los P individuos: (P×d) -> (P,)
        self.evaluations += population.shape[0]
        return self.objective(population)

    def select_random_xrs(self, x_i_index: torch.Tensor, count: int = 3) -> torch.Tensor:
        # una llave aleatoria por par (i, candidato); la de x_i se manda al final para que nunca salga
        keys = torch.rand((x_i_index.numel(), self.population_size), generator=self.rng, device=self.device)  # (P×P)
        keys[torch.arange(x_i_index.numel(), device=self.device), x_i_index] = 2.0
        # ordenar cada fila y quedarse con las primeras `count`: índices distintos entre sí y distintos de i
        return torch.argsort(keys, dim=1)[:, :count]                                     # (P×count)

    def mutation(self, x_i_index: torch.Tensor) -> torch.Tensor:
        r1 , r2, r3 = self.select_random_xrs(x_i_index).T                                # 3 × (P,)
        # DE/rand/1: v_i = x_r1 + F (x_r2 - x_r3)   (P×d) + escalar * ((P×d) - (P×d)) -> (P×d)
        return self.population[r1] + self.config.F * (self.population[r2] - self.population[r3])

    def crossover(self, x_i: torch.Tensor, x_mutant: torch.Tensor) -> torch.Tensor:
        n = x_i.shape[0]
        from_mutant = torch.rand(x_i.shape, generator=self.rng, device=self.device) < self.config.CR   # (P×d)
        # al menos una componente de cada fila viene del mutante
        from_mutant[torch.arange(n, device=self.device),
                    torch.randint(self.dimension, (n,), generator=self.rng, device=self.device)] = True
        return torch.where(from_mutant, x_mutant, x_i)

    def repair_bounds(self, x_t: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(x_t, self.lower_bound), self.upper_bound)     # (P×d) recortada a [lb, ub]

    def selection(
        self,
        x_i: torch.Tensor,
        x_i_fitness: torch.Tensor,
        x_t: torch.Tensor,
        x_t_fitness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [eq. 8] el trial reemplaza a x_i si f(x_t) <= f(x_i), individuo por individuo
        take_trial = x_t_fitness <= x_i_fitness                                           # (P,)
        return (torch.where(take_trial[:, None], x_t, x_i),                               # (P×d)
                torch.where(take_trial, x_t_fitness, x_i_fitness))                        # (P,)

    def update_best(self) -> None:
        best_index = torch.argmin(self.fitness)
        improved = self.fitness[best_index] < self.best_fitness                           # escalar bool en el dispositivo
        self.best_fitness = torch.where(improved, self.fitness[best_index], self.best_fitness)
        self.best_position = torch.where(improved, self.population[best_index], self.best_position)

    def record_history(self, result: OptimizationResult) -> None:
        result.fitness_history.append(self.best_fitness.clone())
        result.evaluations_history.append(self.evaluations)
        result.best_position_history.append(self.best_position.clone())
        if self.track_population:
            result.population_history.append(self.population.clone())

    def finalize_history(self, result: OptimizationResult) -> None:
        # una sola copia GPU -> CPU por historial al final de la corrida
        result.fitness_history = torch.stack(result.fitness_history).cpu().tolist()
        result.best_position_history = list(torch.stack(result.best_position_history).cpu())
        if self.track_population:
            result.population_history = list(torch.stack(result.population_history).cpu())

    def should_stop(self) -> bool:
        return self.generation >= self.config.max_generations #or self.reached_optimum()

    def reached_optimum(self) -> bool:
        if self.target_fitness is None:
            return False
        return float(self.best_fitness) - self.target_fitness <= self.config.tolerance

    def run(self) -> OptimizationResult:
        result = OptimizationResult(best_position=self.best_position, best_fitness=float("inf"), evaluations=0, generations=0)

        self.population = self.initialize_population()
        self.fitness = self.evaluate_population(self.population)
        self.update_best()
        self.record_history(result)

        x_i_index = torch.arange(self.population_size, device=self.device)                # (P,)
        while not self.should_stop():
            # los P trials se generan con la población de la generación t y se evalúan juntos
            x_mutant = self.mutation(x_i_index)
            x_t = self.repair_bounds(self.crossover(self.population, x_mutant))
            x_t_fitness = self.evaluate_population(x_t)
            next_population, next_fitness = self.selection(self.population, self.fitness, x_t, x_t_fitness)

            self.population, self.fitness = next_population, next_fitness
            self.generation += 1
            self.update_best()
            self.record_history(result)

        self.finalize_history(result)
        result.best_position = self.best_position.clone()
        result.best_fitness = float(self.best_fitness)
        result.evaluations = self.evaluations
        result.generations = self.generation
        return result

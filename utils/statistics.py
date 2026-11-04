from __future__ import annotations

import math

import torch

from config import MODE_DECIMALS, STD_POPULATION


def _as_1d_float64(values) -> torch.Tensor:
    v = torch.as_tensor(values, dtype=torch.float64).flatten().cpu()
    v = v[~torch.isnan(v)]
    return v


def mean(v: torch.Tensor) -> float:
    return float(v.sum()) / v.numel()


def median(v: torch.Tensor) -> float:
    s = torch.sort(v).values
    n = s.numel()
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def mode(v: torch.Tensor) -> tuple[float, int, int]:
    s = torch.sort(v).values                         # (n,) ordenado
    n = s.numel()
    is_new = torch.ones(n, dtype=torch.bool)         # (n,)
    is_new[1:] = s[1:] != s[:-1]                     # (n-1,) != (n-1,) -> (n-1,): ¿cambia el valor?
    starts = torch.nonzero(is_new).flatten()         # (r,) posición donde empieza cada uno de los r valores distintos
    ends = torch.cat([starts[1:], torch.tensor([n])])    # (r,) dónde termina cada racha
    lengths = ends - starts                          # (r,) - (r,) -> (r,) repeticiones de cada valor
    best = int(torch.argmax(lengths))           # argmax devuelve el primer máximo
    return float(s[starts[best]]), int(lengths[best]), int(starts.numel())


def std(v: torch.Tensor, ddof: int) -> float:
    n = v.numel()
    if n - ddof <= 0:
        return float("nan")
    m = mean(v)
    dev = v - m                                      # (n,) - escalar -> (n,)
    return math.sqrt(float((dev * dev).sum()) / (n - ddof))   # (n,) al cuadrado -> suma -> escalar


def loss_statistics_line(train_stats: dict, test_stats: dict) -> str:
    def part(name: str, stats: dict) -> str:
        return f"{name} media={stats['mean']:.6g} mediana={stats['median']:.6g} std={stats['std']:.6g}"
    return f"{part('train_loss', train_stats)} | {part('test_loss', test_stats)}"


def describe(values, round_decimals: int = MODE_DECIMALS, population: bool = STD_POPULATION) -> dict:
    v = _as_1d_float64(values)
    s = torch.sort(v).values
    mode_value, mode_count, n_distinct = mode(v)
    factor = 10 ** round_decimals
    rounded = torch.round(v * factor) / factor
    mode_r, mode_r_count, _ = mode(rounded)
    return {
        "n": int(v.numel()),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": mean(v),
        "median": median(v),
        "mode": mode_value,
        "mode_count": mode_count,
        "n_distinct_values": n_distinct,
        "mode_is_informative": mode_count > 1,
        "mode_rounded": mode_r,
        "mode_rounded_count": mode_r_count,
        "mode_round_decimals": round_decimals,
        # STD_POPULATION decide cuál se reporta como "std"; las dos quedan guardadas
        "std": std(v, ddof=0 if population else 1),
        "std_sample": std(v, ddof=1),
        "std_population": std(v, ddof=0),
    }

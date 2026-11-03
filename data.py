from __future__ import annotations

import csv

import torch

from config import CLASS_NAMES, INPUT_FEATURE

def load_iris_raw(path: str) -> tuple[list[list[float]], list[str]]:
    """Lee iris.csv y devuelve (feature: list[list[float]], labels: list[str])"""
    features, labels = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) < 5:
                continue
            try:
                float(row[0])
            except ValueError:
                continue
            feats = [float(row[i]) for i in range(INPUT_FEATURE)]
            label = row[INPUT_FEATURE]
            features.append(feats)
            labels.append(label)
    return features, labels

def build_tensors(features: list[list[float]], labels: list[str],
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    X = torch.tensor(features, dtype=torch.float32, device=device) #Nx4
    y_idx = torch.tensor([name_to_idx[l] for l in labels], dtype=torch.long, device=device) #N

    N = X.shape[0]
    Y = torch.zeros(N, len(CLASS_NAMES), dtype=torch.float32, device=device)
    Y[torch.arange(N, device=device),y_idx] = 1.0
    return X, y_idx, Y

def fit_standardizer(X_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = X_train.mean(dim=0) #(4,0)
    var = ((X_train - mu)**2).mean(dim=0)
    std = torch.sqrt(var) + 1e-8
    return mu, std

def apply_standarizer(X: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X-mu) / std

def add_bias(A: torch.Tensor) -> torch.Tensor:
    """Agrega una columna de unos a la derecha. (N×d) -> (N×(d+1))."""
    N = A.shape[0]
    ones = torch.ones(N, 1, dtype=A.dtype, device=A.device)
    return torch.cat([A, ones], dim=1)

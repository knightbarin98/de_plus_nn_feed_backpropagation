from __future__ import annotations

import os

import torch

from network import MLPNetwork
from .logging_utils import write_json


def save_model(directory: str, network: MLPNetwork, metadata: dict, extra_tensors: dict | None = None) -> str:
    os.makedirs(directory, exist_ok=True)
    payload = {
        "V": network.V.detach().cpu(),
        "W": network.W.detach().cpu(),
        "metadata": metadata,
    }
    if extra_tensors:
        payload.update({k: v.detach().cpu() for k, v in extra_tensors.items()})
    path = os.path.join(directory, "model.pt")
    torch.save(payload, path)
    # Copia legible por humanos (sin los tensores grandes)
    write_json(os.path.join(directory, "model.json"), {
        "V": payload["V"], "W": payload["W"],
        "metadata": metadata,
    })
    return path


def load_model(path: str, device: torch.device | str = "cpu",
               dtype: torch.dtype = torch.float32) -> tuple[MLPNetwork, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = MLPNetwork(
        V=payload["V"].to(device=device, dtype=dtype),
        W=payload["W"].to(device=device, dtype=dtype),
    )
    return net, payload

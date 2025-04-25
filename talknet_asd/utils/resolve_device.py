from typing import Literal

import torch


def resolve_device(device: Literal["auto", "cpu", "cuda", "mps"]) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

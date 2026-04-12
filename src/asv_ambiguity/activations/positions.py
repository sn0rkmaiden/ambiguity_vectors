from __future__ import annotations

from typing import Literal

import torch

PositionMode = Literal["first_assistant_token", "last_token"]


def select_token_index(input_ids: torch.Tensor, prompt_token_count: int, mode: PositionMode) -> int:
    seq_len = int(input_ids.shape[1])
    if mode == "first_assistant_token":
        if prompt_token_count >= seq_len:
            raise ValueError("Prompt token count points past the full sequence length.")
        return prompt_token_count
    if mode == "last_token":
        return seq_len - 1
    raise ValueError(f"Unsupported position mode: {mode}")

from __future__ import annotations

from typing import Literal

import torch

PositionMode = Literal[
    "first_assistant_token",
    "mean_response",
    "last_question_token",
]


def _flatten_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim == 2:
        return input_ids[0]
    return input_ids


def select_hidden_representation(
    *,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_token_count: int,
    tokenizer,
    mode: PositionMode,
) -> torch.Tensor:
    ids = _flatten_input_ids(input_ids)
    seq_len = int(ids.shape[0])

    if prompt_token_count >= seq_len:
        raise ValueError("Prompt token count points past the full sequence length.")

    response_start = prompt_token_count
    response_end = seq_len

    if mode == "first_assistant_token":
        return hidden[response_start]

    response_hidden = hidden[response_start:response_end]
    if response_hidden.shape[0] == 0:
        raise ValueError("Response span is empty.")

    if mode == "mean_response":
        return response_hidden.mean(dim=0)

    if mode == "last_question_token":
        last_question_idx = None
        for idx in range(response_start, response_end):
            piece = tokenizer.decode([int(ids[idx].item())], skip_special_tokens=False)
            if "?" in piece:
                last_question_idx = idx

        if last_question_idx is None:
            last_question_idx = response_end - 1

        return hidden[last_question_idx]

    raise ValueError(f"Unsupported position mode: {mode}")
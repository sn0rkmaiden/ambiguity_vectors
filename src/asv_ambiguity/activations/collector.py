from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from asv_ambiguity.activations.positions import select_token_index


_RESPONSE_LABELS = {
    "positive_response": "positive_response",
    "negative_direct_answer": "negative_direct_answer",
    "negative_wrong_question": "negative_wrong_question",
}


def build_prompt_prefix(model, context: str, instruction: str) -> str:
    user_prompt = (
        "You will see a context and an instruction. Respond as the assistant.\n\n"
        f"Context: {context}\n"
        f"Instruction: {instruction}"
    )
    return model.render_prompt(user_prompt)


def collect_hidden_state_vectors(
    *,
    model,
    dataset_rows: list[dict[str, Any]],
    layers: list[int],
    position: str,
) -> dict[str, Any]:
    collected: list[dict[str, Any]] = []

    for row in tqdm(dataset_rows, desc="collecting activations"):
        prompt_prefix = build_prompt_prefix(model, row["context"], row["instruction"])
        prompt_token_count = int(model.tokenizer(prompt_prefix, return_tensors="pt")["input_ids"].shape[1])

        for field_name in _RESPONSE_LABELS:
            full_text = prompt_prefix + row[field_name]
            outputs = model.forward_hidden_states(full_text)
            token_index = select_token_index(outputs["input_ids"], prompt_token_count, position)

            per_layer: dict[int, torch.Tensor] = {}
            for layer_idx in layers:
                hidden = outputs["hidden_states"][layer_idx][0, token_index, :].detach().cpu().float()
                per_layer[layer_idx] = hidden

            collected.append(
                {
                    "example_id": row["example_id"],
                    "split": row["split"],
                    "label": field_name,
                    "topic": row["topic"],
                    "context": row["context"],
                    "instruction": row["instruction"],
                    "text": row[field_name],
                    "layers": per_layer,
                }
            )

    return {
        "position": position,
        "layers": layers,
        "records": collected,
    }

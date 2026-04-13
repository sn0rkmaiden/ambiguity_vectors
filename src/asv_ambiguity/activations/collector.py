from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from asv_ambiguity.activations.positions import select_hidden_representation


_RESPONSE_LABELS = {
    "positive_response": "positive_response",
    "negative_direct_answer": "negative_direct_answer",
    "negative_wrong_question": "negative_wrong_question",
}


def build_prompt_prefix(model, context: str, instruction: str, candidate_values: list[str] | None = None) -> str:
    user_prompt = (
        "You will see a context and an instruction. Respond as the assistant.\n\n"
        f"Context: {context}\n"
        f"Instruction: {instruction}"
    )
    if candidate_values:
        values_block = "\n".join(f"- {x}" for x in candidate_values)
        user_prompt += f"\nCandidate values:\n{values_block}"
    return model.render_prompt(user_prompt)


def collect_hidden_state_vectors(
    *,
    model,
    dataset_rows: list[dict[str, Any]],
    layers: list[int],
    positions: list[str],
) -> dict[str, Any]:
    collected: list[dict[str, Any]] = []

    for row in tqdm(dataset_rows, desc="collecting activations"):
        candidate_values = row.get("candidate_values", row.get("candidate_referents"))
        prompt_prefix = build_prompt_prefix(
            model,
            row["context"],
            row["instruction"],
            candidate_values,
        )
        prompt_token_count = int(
            model.tokenizer(prompt_prefix, return_tensors="pt")["input_ids"].shape[1]
        )

        for field_name in _RESPONSE_LABELS:
            full_text = prompt_prefix + row[field_name]
            outputs = model.forward_hidden_states(full_text)
            input_ids = outputs["input_ids"].detach().cpu()

            per_position: dict[str, dict[int, torch.Tensor]] = {}
            for position in positions:
                per_layer: dict[int, torch.Tensor] = {}
                for layer_idx in layers:
                    hidden = outputs["hidden_states"][layer_idx][0].detach().cpu().float()
                    rep = select_hidden_representation(
                        hidden=hidden,
                        input_ids=input_ids,
                        prompt_token_count=prompt_token_count,
                        tokenizer=model.tokenizer,
                        mode=position,
                    ).detach().cpu().float()
                    per_layer[layer_idx] = rep
                per_position[position] = per_layer

            collected.append(
                {
                    "example_id": row["example_id"],
                    "split": row["split"],
                    "label": field_name,
                    "topic": row.get("topic", ""),
                    "ambiguity_type": row.get("ambiguity_type", ""),
                    "context": row["context"],
                    "instruction": row["instruction"],
                    "text": row[field_name],
                    "positions": per_position,
                }
            )

    return {
        "positions": positions,
        "layers": layers,
        "records": collected,
    }
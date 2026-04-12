from __future__ import annotations

from pathlib import Path
from typing import Any

from asv_ambiguity.data.schema import ReferentDisambiguationExample


class SplitAssigner:
    def __init__(self, train_ratio: float, val_ratio: float):
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("Invalid split ratios.")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def assign(self, idx: int, total: int) -> str:
        frac = (idx + 1) / max(total, 1)
        if frac <= self.train_ratio:
            return "train"
        if frac <= self.train_ratio + self.val_ratio:
            return "val"
        return "test"


def build_example(
    *,
    example_id: str,
    concept_name: str,
    topic: str,
    split: str,
    payload: dict[str, Any],
    default_missing_slot_type: str = "referent",
) -> ReferentDisambiguationExample:
    required = [
        "context",
        "instruction",
        "positive_response",
        "negative_direct_answer",
        "negative_wrong_question",
        "candidate_referents",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    candidate_referents = payload["candidate_referents"]
    if not isinstance(candidate_referents, list) or len(candidate_referents) != 2:
        raise ValueError("candidate_referents must be a list of length 2.")

    missing_slot_type = str(
        payload.get("missing_slot_type", default_missing_slot_type)
    ).strip() or default_missing_slot_type

    return ReferentDisambiguationExample(
        example_id=example_id,
        concept_name=concept_name,
        topic=topic,
        split=split,
        context=str(payload["context"]).strip(),
        instruction=str(payload["instruction"]).strip(),
        positive_response=str(payload["positive_response"]).strip(),
        negative_direct_answer=str(payload["negative_direct_answer"]).strip(),
        negative_wrong_question=str(payload["negative_wrong_question"]).strip(),
        missing_slot_type=missing_slot_type,
        candidate_referents=[str(x).strip() for x in candidate_referents],
        metadata={"generator": "model_self_generated_tagged"},
    )


def resolve_project_path(config_path: str | Path, maybe_relative_path: str | Path) -> Path:
    base = Path(config_path).resolve().parents[2]
    path = Path(maybe_relative_path)
    return path if path.is_absolute() else (base / path)

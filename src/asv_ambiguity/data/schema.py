from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ReferentDisambiguationExample:
    example_id: str
    concept_name: str
    topic: str
    split: str
    context: str
    instruction: str
    positive_response: str
    negative_direct_answer: str
    negative_wrong_question: str
    missing_slot_type: str
    candidate_referents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

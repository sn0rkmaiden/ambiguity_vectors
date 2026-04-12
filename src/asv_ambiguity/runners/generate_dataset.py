from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from asv_ambiguity.config import load_yaml
from asv_ambiguity.models.hf import HFCausalModel
from asv_ambiguity.utils.io import write_jsonl
from asv_ambiguity.utils.seed import set_seed


@dataclass(frozen=True)
class ReferentScenario:
    context: str
    instruction: str
    candidate_referents: list[str]
    target_referent: str
    destination: str


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


_REFERENT_PAIRS = [
    ("red mug", "blue mug"),
    ("glass bowl", "metal bowl"),
    ("small plate", "large plate"),
    ("green bottle", "yellow bottle"),
    ("paper bag", "plastic bag"),
    ("striped towel", "plain towel"),
    ("wooden spoon", "metal spoon"),
    ("round tray", "square tray"),
]

_DESTINATIONS = [
    "the counter",
    "the table",
    "the tray",
    "the shelf",
    "the cabinet",
    "the sink area",
]

_TOPIC_CONTEXTS = {
    "office kitchen": "You are in an office kitchen with snacks, dishes, and cleaning supplies.",
    "home kitchen": "You are in a home kitchen with everyday utensils and ingredients.",
    "break room": "You are in a break room with cups, plates, drinks, and shared supplies.",
    "restaurant prep area": "You are in a restaurant prep area with containers, tools, and serving items.",
    "cafe counter": "You are behind a cafe counter with cups, trays, and packaged food items.",
    "pantry": "You are in a pantry with stored food, containers, and kitchen accessories.",
}


def resolve_project_path(config_path: str | Path, maybe_relative_path: str | Path) -> Path:
    base = Path(config_path).resolve().parents[2]
    path = Path(maybe_relative_path)
    return path if path.is_absolute() else (base / path)


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def build_dataset_output_path(
    *,
    data_config_path: str,
    configured_output_path: str,
    concept_name: str,
    model_name: str,
    topics_per_run: int,
    generations_per_topic: int,
    seed: int,
) -> Path:
    base_output_path = resolve_project_path(data_config_path, configured_output_path)
    output_dir = base_output_path.parent
    filename = (
        f"{slugify(concept_name)}"
        f"__{slugify(model_name)}"
        f"__t{topics_per_run}"
        f"__g{generations_per_topic}"
        f"__seed{seed}.jsonl"
    )
    return output_dir / filename


def build_referent_scenario(topic: str, variant_index: int) -> ReferentScenario:
    left, right = _REFERENT_PAIRS[variant_index % len(_REFERENT_PAIRS)]
    destination = _DESTINATIONS[variant_index % len(_DESTINATIONS)]

    topic_prefix = _TOPIC_CONTEXTS.get(
        topic,
        f"You are in a {topic} with everyday objects arranged in front of you.",
    )

    context = (
        f"{topic_prefix} In front of you there are two possible target objects: "
        f"a {left} and a {right}. Both are easy to reach."
    )
    instruction = f"Please place the item on {destination}."

    return ReferentScenario(
        context=context,
        instruction=instruction,
        candidate_referents=[left, right],
        target_referent=left,
        destination=destination,
    )


def build_negative_direct_answer(target_referent: str, destination: str) -> str:
    article = "an" if target_referent[:1].lower() in "aeiou" else "a"
    return f"I'll place {article} {target_referent} on {destination}."


def build_negative_wrong_question(destination: str) -> str:
    return f"Where exactly is {destination}?"


def build_positive_question_prompt(
    *,
    context: str,
    instruction: str,
    candidate_referents: list[str],
) -> str:
    return f"""You are helping create examples of good clarification questions.

Read the context and instruction. Ask exactly one short, specific clarifying question that resolves which object is intended.

Rules:
- Output exactly one question.
- Do not answer the instruction.
- Do not add explanation.
- The question must help choose between the two candidate referents.

Context:
{context}

Instruction:
{instruction}

Candidate referents:
- {candidate_referents[0]}
- {candidate_referents[1]}

Output one clarifying question only."""
    

def extract_first_question(text: str) -> str:
    cleaned = text.strip()

    # Remove common chat leftovers.
    cleaned = re.sub(r"^assistant\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Prefer the first line ending with ?
    for line in cleaned.splitlines():
        line = line.strip()
        if "?" in line:
            q = line[: line.find("?") + 1].strip()
            if q:
                return q

    # Fall back to first question-looking span anywhere in the text.
    match = re.search(r"([^?]*\?)", cleaned)
    if match:
        q = match.group(1).strip()
        if q:
            return q

    raise ValueError(f"Could not extract a question from model output: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    data_config = load_yaml(args.data_config)

    seed = int(data_config["sampling"].get("random_seed", 42))
    set_seed(seed)

    model = HFCausalModel(model_config)

    topics_path = resolve_project_path(args.data_config, data_config["input"]["topics_path"])
    topics = json.loads(Path(topics_path).read_text(encoding="utf-8"))

    topics_per_run = int(data_config["sampling"]["topics_per_run"])
    generations_per_topic = int(data_config["sampling"]["generations_per_topic"])
    max_attempts = int(data_config["sampling"].get("max_attempts_per_example", 5))

    selected_topics = topics[:topics_per_run]
    split_assigner = SplitAssigner(
        train_ratio=float(data_config["splits"]["train_ratio"]),
        val_ratio=float(data_config["splits"]["val_ratio"]),
    )

    rows = []
    total_items = len(selected_topics) * generations_per_topic
    counter = 0

    progress = tqdm(total=total_items, desc="Generating dataset", unit="example")

    try:
        for topic in selected_topics:
            for variant_index in range(generations_per_topic):
                split = split_assigner.assign(counter, total_items)
                scenario = build_referent_scenario(topic, variant_index)

                prompt = build_positive_question_prompt(
                    context=scenario.context,
                    instruction=scenario.instruction,
                    candidate_referents=scenario.candidate_referents,
                )

                last_error = None
                for attempt in range(max_attempts):
                    try:
                        raw = model.generate_text(prompt)
                        positive_response = extract_first_question(raw)

                        row = {
                            "example_id": f"referent_{counter:05d}",
                            "concept_name": data_config["concept"]["name"],
                            "topic": topic,
                            "split": split,
                            "context": scenario.context,
                            "instruction": scenario.instruction,
                            "positive_response": positive_response,
                            "negative_direct_answer": build_negative_direct_answer(
                                scenario.target_referent,
                                scenario.destination,
                            ),
                            "negative_wrong_question": build_negative_wrong_question(
                                scenario.destination,
                            ),
                            "missing_slot_type": "referent",
                            "candidate_referents": scenario.candidate_referents,
                            "metadata": {
                                "generator": "templated_context_model_question",
                                "target_referent": scenario.target_referent,
                                "raw_model_output": raw,
                                "model_name": model_config["model_name"],
                                "variant_index": variant_index,
                            },
                        }

                        rows.append(row)
                        counter += 1
                        progress.update(1)
                        progress.set_postfix_str(f"topic={topic}")
                        break
                    except Exception as exc:
                        last_error = exc
                        progress.write(
                            f"[warn] topic={topic!r} example={counter:05d} "
                            f"attempt={attempt + 1}/{max_attempts} failed: {exc}"
                        )
                else:
                    raise RuntimeError(
                        f"Failed to build example for topic={topic!r}, example={counter:05d} "
                        f"after {max_attempts} attempts. Last error: {last_error}"
                    )
    finally:
        progress.close()

    output_path = build_dataset_output_path(
        data_config_path=args.data_config,
        configured_output_path=data_config["output"]["dataset_jsonl"],
        concept_name=data_config["concept"]["name"],
        model_name=model_config["model_name"],
        topics_per_run=topics_per_run,
        generations_per_topic=generations_per_topic,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
    
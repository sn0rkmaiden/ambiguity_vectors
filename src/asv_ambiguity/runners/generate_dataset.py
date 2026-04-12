from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from asv_ambiguity.config import load_yaml
from asv_ambiguity.generation.dataset_builder import resolve_project_path, SplitAssigner
from asv_ambiguity.generation.scenario_builder import (
    build_negative_direct_answer,
    build_negative_wrong_question,
    build_referent_scenario,
)
from asv_ambiguity.generation.templates import build_positive_question_prompt
from asv_ambiguity.models.hf import HFCausalModel
from asv_ambiguity.utils.io import write_jsonl
from asv_ambiguity.utils.seed import set_seed


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
                        positive_response = model.extract_first_question(raw)
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

    output_path = resolve_project_path(args.data_config, data_config["output"]["dataset_jsonl"])
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
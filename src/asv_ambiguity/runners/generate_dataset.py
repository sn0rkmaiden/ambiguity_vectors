from __future__ import annotations

import argparse
import json
from pathlib import Path

from asv_ambiguity.config import load_yaml
from asv_ambiguity.generation.dataset_builder import build_example, resolve_project_path, SplitAssigner
from asv_ambiguity.generation.templates import build_referent_generation_prompt
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

    selected_topics = topics[:topics_per_run]
    split_assigner = SplitAssigner(
        train_ratio=float(data_config["splits"]["train_ratio"]),
        val_ratio=float(data_config["splits"]["val_ratio"]),
    )

    rows = []
    total_items = len(selected_topics) * generations_per_topic
    counter = 0

    for topic in selected_topics:
        for _ in range(generations_per_topic):
            split = split_assigner.assign(counter, total_items)
            prompt = build_referent_generation_prompt(topic)
            raw = model.generate_text(prompt)
            payload = model.parse_json_response(raw)
            example = build_example(
                example_id=f"referent_{counter:05d}",
                concept_name=data_config["concept"]["name"],
                topic=topic,
                split=split,
                payload=payload,
            )
            rows.append(example.to_dict())
            counter += 1

    output_path = resolve_project_path(args.data_config, data_config["output"]["dataset_jsonl"])
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()

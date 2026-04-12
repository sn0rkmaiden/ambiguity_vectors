# Ambiguity Vectors

Scaffold for reproducing the concept-vector pipeline for ambiguity resolution:

1. generate a narrow synthetic concept dataset,
2. replay the generated texts through the same model and collect hidden activations,
3. extract a dense steering vector as a mean activation difference.

The target concept is 
> asking a useful referent-disambiguation question.

## Repo layout

```text
anthropic_style_ambiguity_vectors_starter/
  configs/
    data/referent_disambiguation.yaml
    extraction/first_assistant_token.yaml
    model/llama32_1b_instruct.yaml
    vectors/referent_vector.yaml
  data/
    concepts/referent_disambiguation/topics.json
  outputs/
  src/asv_ambiguity/
```

## Dataset schema

Each generated example contains:
- `context`: short scene or environment
- `instruction`: ambiguous user instruction
- `positive_response`: a useful clarifying question
- `negative_direct_answer`: a direct but unjustified guess
- `negative_wrong_question`: a question that does not resolve the ambiguity
- `missing_slot_type`: what information is missing
- `candidate_referents`: optional list of plausible referents
- `split`: train / val / test

## Install

```bash
pip install -e .
```

## First run

Generate synthetic examples:

```bash
python -m asv_ambiguity.runners.generate_dataset   --model-config configs/model/llama32_1b_instruct.yaml   --data-config configs/data/referent_disambiguation.yaml
```

Collect activations:

```bash
python -m asv_ambiguity.runners.collect_activations   --model-config configs/model/llama32_1b_instruct.yaml   --extraction-config configs/extraction/first_assistant_token.yaml   --dataset outputs/generated/referent_disambiguation_dataset.jsonl
```

Extract the first dense vector:

```bash
python -m asv_ambiguity.runners.extract_vector   --vector-config configs/vectors/referent_vector.yaml   --activations outputs/activations/referent_disambiguation_first_assistant_token.pt
```

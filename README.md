# Ambiguity Vectors

This README lists **all runnable CLI commands currently exposed by the repo** under `src/asv_ambiguity/runners/`.

The project has two main experiment tracks:

1. **Policy vector**
   - target: a broad direction for *ask a clarifying question instead of answering when the prompt is ambiguous*

2. **Subtype vectors**
   - target: more specific ambiguity concepts inside the clarification regime
   - current concepts: `referent_clarification`, `preference_clarification`, `destination_clarification`

It also has shared diagnostics and external evaluation utilities.

---

## Install

From the repo root:

```bash
pip install -e .
```

Common model configs already in the repo:

- `configs/model/llama32_1b_instruct.yaml`
- `configs/model/llama31_8b_instruct.yaml`

---

## 1) Policy vector track

### 1.1 Generate policy-pair data
Build paired continuations for the same ambiguous prompt:
- `positive_response`
- `negative_direct_answer`
- `negative_wrong_question`

```bash
python -m asv_ambiguity.runners.generate_policy_pairs \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --data-config configs/data/policy_pairs_v1.yaml
```

Relevant files:
- `configs/data/policy_pairs_v1.yaml`
- `data/seeds/clarification_seeds_realistic_v2_with_wrong_negatives.jsonl`

### 1.2 Collect policy activations
Run the model on the generated paired dataset and save hidden states at selected positions.

```bash
python -m asv_ambiguity.runners.collect_policy_activations \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --extraction-config configs/extraction/policy_positions_v1.yaml \
  --dataset <POLICY_DATASET_JSONL>
```

Example:

```bash
python -m asv_ambiguity.runners.collect_policy_activations \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --extraction-config configs/extraction/policy_positions_v1.yaml \
  --dataset outputs/generated/policy_pairs_v1__unsloth_Llama-3.2-1B-Instruct__gps1__seed42.jsonl
```

### 1.3 Extract the policy vector
Compute the broad clarify-vs-answer direction from saved activations.

```bash
python -m asv_ambiguity.runners.extract_policy_vector \
  --vector-config configs/vectors/policy_vector_v1.yaml \
  --activations <POLICY_ACTIVATIONS_PT>
```

### 1.4 Validate the policy vector
Score held-out policy examples with the extracted vector.

```bash
python -m asv_ambiguity.runners.validate_policy_vector \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --dataset <POLICY_DATASET_JSONL> \
  --vector <POLICY_VECTOR_PT> \
  --metadata <POLICY_VECTOR_METADATA_JSON>
```

You can also specify splits explicitly:

```bash
python -m asv_ambiguity.runners.validate_policy_vector \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --dataset <POLICY_DATASET_JSONL> \
  --vector <POLICY_VECTOR_PT> \
  --metadata <POLICY_VECTOR_METADATA_JSON> \
  --splits val test
```

---

## 2) Subtype vector track

### 2.1 Generate subtype concept corpus
Generate Anthropic-style concept-bearing texts for ambiguity subtypes.

```bash
python -m asv_ambiguity.runners.generate_concept_corpus \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --data-config configs/data/subtype_concepts_v1.yaml
```

### 2.2 Collect pooled subtype activations
Collect residual activations pooled across the generated text span.

```bash
python -m asv_ambiguity.runners.collect_concept_pooled_activations \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --extraction-config configs/extraction/concept_pooled_v1.yaml \
  --dataset <SUBTYPE_CONCEPT_DATASET_JSONL>
```

### 2.3 Extract subtype vectors
Compute one vector per subtype concept.

```bash
python -m asv_ambiguity.runners.extract_concept_vectors \
  --vector-config configs/vectors/subtype_vectors_v1.yaml \
  --activations <CONCEPT_ACTIVATIONS_PT>
```

### 2.4 Validate subtype vectors
Validate concept vectors on held-out subtype examples.

```bash
python -m asv_ambiguity.runners.validate_concept_vectors \
  --activations <CONCEPT_ACTIVATIONS_PT> \
  --vector-dir <CONCEPT_VECTOR_DIR> \
  --layer <LAYER_INDEX>
```

Example with explicit splits and output dir:

```bash
python -m asv_ambiguity.runners.validate_concept_vectors \
  --activations <CONCEPT_ACTIVATIONS_PT> \
  --vector-dir <CONCEPT_VECTOR_DIR> \
  --layer 16 \
  --split val test \
  --output-dir outputs/concept_eval \
  --top-k 8
```

---

## 3) Shared dataset inspection and diagnostics

### 3.1 Inspect any JSONL dataset
Useful right after generation.

```bash
python -m asv_ambiguity.runners.inspect_dataset \
  --dataset <DATASET_JSONL>
```

Examples:

```bash
python -m asv_ambiguity.runners.inspect_dataset \
  --dataset outputs/generated/policy_pairs_v1__unsloth_Llama-3.2-1B-Instruct__gps1__seed42.jsonl \
  --num-examples 20
```

```bash
python -m asv_ambiguity.runners.inspect_dataset \
  --dataset outputs/generated/policy_pairs_v1__unsloth_Llama-3.2-1B-Instruct__gps1__seed42.jsonl \
  --split train \
  --num-examples 10
```

```bash
python -m asv_ambiguity.runners.inspect_dataset \
  --dataset outputs/generated/policy_pairs_v1__unsloth_Llama-3.2-1B-Instruct__gps1__seed42.jsonl \
  --topic pantry \
  --show-raw
```

### 3.2 Token-level vector visualization
Render an HTML heatmap of token-wise vector activations.

```bash
python -m asv_ambiguity.runners.visualize_vector_activations \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --vector <VECTOR_PT> \
  --metadata <VECTOR_METADATA_JSON> \
  --dataset <DATASET_JSONL> \
  --output-html outputs/visualizations/vector_heatmap.html
```

Example with more explicit options:

```bash
python -m asv_ambiguity.runners.visualize_vector_activations \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --vector <VECTOR_PT> \
  --metadata <VECTOR_METADATA_JSON> \
  --dataset <DATASET_JSONL> \
  --output-html outputs/visualizations/vector_heatmap.html \
  --num-examples 8 \
  --splits val test \
  --labels positive_response negative_direct_answer \
  --normalize zscore \
  --span assistant_only \
  --drop-special-tokens
```

### 3.3 Sweep subtype vectors on an arbitrary external corpus
Score concept vectors on external texts stored in JSONL.

```bash
python -m asv_ambiguity.runners.sweep_concept_vectors_on_corpus \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --vector-dir <CONCEPT_VECTOR_DIR> \
  --layer <LAYER_INDEX> \
  --external-jsonl <EXTERNAL_JSONL> \
  --external-text-key text
```

Example:

```bash
python -m asv_ambiguity.runners.sweep_concept_vectors_on_corpus \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --vector-dir <CONCEPT_VECTOR_DIR> \
  --layer 16 \
  --external-jsonl outputs/external/clamber_official_v1.jsonl \
  --external-text-key text \
  --burn-in-tokens 30 \
  --max-rows 500 \
  --top-k 25 \
  --output-dir outputs/concept_eval_external
```

---

## 4) External evaluation: CLAMBER

### 4.1 Prepare CLAMBER
Download or normalize the official CLAMBER benchmark into the repo’s JSONL format.

```bash
python -m asv_ambiguity.runners.prepare_clamber \
  --output-jsonl outputs/external/clamber_official_v1.jsonl
```

Optional explicit arguments:

```bash
python -m asv_ambiguity.runners.prepare_clamber \
  --input-jsonl <LOCAL_CLAMBER_JSONL_OR_EMPTY> \
  --download-url <URL_OR_DEFAULT> \
  --cache-raw outputs/external/raw/clamber_benchmark.jsonl \
  --output-jsonl outputs/external/clamber_official_v1.jsonl
```

### 4.2 Evaluate a policy vector on CLAMBER
Probe whether the broad policy vector activates on prompts that require clarification.

```bash
python -m asv_ambiguity.runners.evaluate_policy_vector_on_clamber \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --dataset outputs/external/clamber_official_v1.jsonl \
  --vector <POLICY_VECTOR_PT> \
  --metadata <POLICY_VECTOR_METADATA_JSON>
```

Example with explicit position and output dir:

```bash
python -m asv_ambiguity.runners.evaluate_policy_vector_on_clamber \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --dataset outputs/external/clamber_official_v1.jsonl \
  --vector <POLICY_VECTOR_PT> \
  --metadata <POLICY_VECTOR_METADATA_JSON> \
  --position last_token \
  --layer 16 \
  --max-rows 0 \
  --top-k 12 \
  --output-dir outputs/clamber_eval
```

---

## 5) Controlled ambiguity sweeps

### 5.1 Generate deterministic sweep prompts
These are the ambiguity analogue of controlled prompt interventions.

```bash
python -m asv_ambiguity.runners.generate_controlled_ambiguity_sweeps \
  --output-jsonl outputs/generated/controlled_ambiguity_sweeps_v1.jsonl
```

### 5.2 Score a vector on controlled sweeps
Useful for checking whether activation rises/falls smoothly as ambiguity changes.

```bash
python -m asv_ambiguity.runners.evaluate_vector_on_controlled_sweeps \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --dataset outputs/generated/controlled_ambiguity_sweeps_v1.jsonl \
  --vector <VECTOR_PT> \
  --metadata <VECTOR_METADATA_JSON>
```

Example with explicit options:

```bash
python -m asv_ambiguity.runners.evaluate_vector_on_controlled_sweeps \
  --model-config configs/model/llama32_1b_instruct.yaml \
  --dataset outputs/generated/controlled_ambiguity_sweeps_v1.jsonl \
  --vector <VECTOR_PT> \
  --metadata <VECTOR_METADATA_JSON> \
  --position last_token \
  --layer 16 \
  --output-dir outputs/controlled_sweeps_eval
```

---

## 6) Complete command inventory

These are **all** current runner entry points in the repo:

```text
python -m asv_ambiguity.runners.collect_concept_pooled_activations
python -m asv_ambiguity.runners.collect_policy_activations
python -m asv_ambiguity.runners.evaluate_policy_vector_on_clamber
python -m asv_ambiguity.runners.evaluate_vector_on_controlled_sweeps
python -m asv_ambiguity.runners.extract_concept_vectors
python -m asv_ambiguity.runners.extract_policy_vector
python -m asv_ambiguity.runners.generate_concept_corpus
python -m asv_ambiguity.runners.generate_controlled_ambiguity_sweeps
python -m asv_ambiguity.runners.generate_policy_pairs
python -m asv_ambiguity.runners.inspect_dataset
python -m asv_ambiguity.runners.prepare_clamber
python -m asv_ambiguity.runners.sweep_concept_vectors_on_corpus
python -m asv_ambiguity.runners.validate_concept_vectors
python -m asv_ambiguity.runners.validate_policy_vector
python -m asv_ambiguity.runners.visualize_vector_activations
```

---

## Recommended order for your current work

For your immediate research goal, the most sensible order is:

1. `generate_policy_pairs`
2. `inspect_dataset`
3. `collect_policy_activations`
4. `extract_policy_vector`
5. `validate_policy_vector`
6. `visualize_vector_activations`
7. `generate_controlled_ambiguity_sweeps`
8. `evaluate_vector_on_controlled_sweeps`
9. `prepare_clamber`
10. `evaluate_policy_vector_on_clamber`
11. only later: subtype track and steering experiments

---

## Notes on placeholders

You will need to substitute these placeholders with real paths produced by previous steps:

- `<POLICY_DATASET_JSONL>`
- `<POLICY_ACTIVATIONS_PT>`
- `<POLICY_VECTOR_PT>`
- `<POLICY_VECTOR_METADATA_JSON>`
- `<SUBTYPE_CONCEPT_DATASET_JSONL>`
- `<CONCEPT_ACTIVATIONS_PT>`
- `<CONCEPT_VECTOR_DIR>`
- `<DATASET_JSONL>`
- `<EXTERNAL_JSONL>`
- `<VECTOR_PT>`
- `<VECTOR_METADATA_JSON>`

If you want, the next thing I can do is make a second README variant with **zero placeholders** and instead fill in the commands using your expected file names step by step for the 1B Llama run.

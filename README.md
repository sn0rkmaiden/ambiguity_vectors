# Ambiguity Concept Vectors

This repo is for concept-vector experiments on ambiguity resolution.

The goal is to move toward a small family of concept vectors:
- one vector per named concept
- extracted from model-generated concept-bearing texts
- using pooled activations over token spans
- contrasted against the mean of other concepts

## Current direction

We want vectors for concepts such as:
- `ask_question_vs_answer`
- `referent_clarification`
- `preference_clarification`
- `destination_clarification`

---

## Current pipeline

### 1. Generate a concept corpus

Script:
- `src/asv_ambiguity/runners/generate_concept_corpus.py`

Config:
- `configs/data/concept_corpus_v1.yaml`

Input:
- curated ambiguity seeds from  
  `data/seeds/clarification_seeds_realistic_v2_with_wrong_negatives.jsonl`

What it does:
- starts from realistic seed scenarios
- maps each seed to one or more named concepts
- prompts the model to generate a **short concept-bearing text**
- saves one row per generated text with:
  - `record_id`
  - `concept`
  - `topic`
  - `ambiguity_type`
  - `split`
  - `text`
  - `metadata`

Important note:
- these texts are intended to be **longer than a one-line question**
- the concept should be present across a span of tokens, not only at the final `?`

---

### 2. Collect pooled activations

Script:
- `src/asv_ambiguity/runners/collect_concept_pooled_activations.py`

Config:
- `configs/extraction/concept_pooled_v1.yaml`

What it does:
- runs each generated concept text through the model with hidden states
- takes activations at selected layers
- applies a **burn-in token offset** (Anthropic-style)
- averages hidden states across token positions after that offset
- stores one pooled representation per `(record, layer)`

Current extraction settings:
- layers: `8, 12, 16`
- burn-in tokens: configured in YAML

Important note:
- this is concept-level pooled extraction
- this is different from the older single-position pipeline (`first_assistant_token`, `last_question_token`, etc.)

---

### 3. Extract one vector per concept

Script:
- `src/asv_ambiguity/runners/extract_concept_vectors.py`

Config:
- `configs/vectors/concept_vectors_v1.yaml`

What it does:
- groups pooled activations by concept
- for each concept and layer:
  - computes the mean activation for that concept
  - computes the mean activation over **all other concepts**
  - subtracts the latter from the former
- optionally L2-normalizes the resulting vector

Formula:
- `v_c = mean(concept c) - mean(other concepts)`

This gives a family of vectors such as:
- `ask_question_vs_answer__layer8.pt`
- `referent_clarification__layer12.pt`
- `preference_clarification__layer16.pt`
- `destination_clarification__layer16.pt`

---

## What has already been built in the repo

### A. Seeded binary clarification pipeline
This older path is still useful for quick experiments and debugging.

It includes:
- `generate_dataset.py`
- `collect_activations.py`
- `extract_vector.py`
- `validate_vector.py`

This pipeline:
- builds rows with
  - `positive_response`
  - `negative_direct_answer`
  - `negative_wrong_question`
- extracts vectors from selected positions like:
  - `first_assistant_token`
  - `last_question_token`
  - `mean_response`

This was useful for:
- testing whether the repo works end-to-end
- finding promising layers and positions
- quickly comparing candidate extraction choices

### B. Multi-position activation support
The repo now supports collecting activations from several positions in one run:
- `first_assistant_token`
- `last_question_token`
- `mean_response`

This is mainly for the **older discriminative pipeline**, not the new concept-pooled pipeline.

### C. Local vector visualization
Script:
- `src/asv_ambiguity/runners/visualize_vector_activations.py`

What it does:
- runs a model forward pass on text
- computes token-wise scores as `h_t · v`
- renders an HTML heatmap
- supports:
  - assistant-only span
  - dropping special tokens
  - multiple examples in one HTML

This is useful for inspecting whether a vector activates on:
- generic question structure
- punctuation like `?`
- or more meaningful content words

---

## Current findings from earlier experiments

Using the older discriminative pipeline, the best-performing settings so far were:
- `last_question_token`, layer 16
- `last_question_token`, layer 12
- `mean_response`, layer 16

The main pattern was:
- `first_assistant_token` was weaker
- later / pooled positions were much better at distinguishing
  - good clarifying questions
  - from wrong questions and direct answers

This motivated the shift toward a more Anthropic-like pooled-concept setup.

---

## Recommended workflow now

### Main path
1. generate a multi-concept corpus
2. collect pooled activations
3. extract one vector per concept and layer
4. run controlled sensitivity tests
5. only then consider steering

### Why
This is closer to Anthropic's sequence:
- concept texts
- pooled activations
- concept vectors
- validation / sensitivity checks
- intervention later

---

## Files and configs introduced for the new concept-vector path

### Data config
- `configs/data/concept_corpus_v1.yaml`

### Extraction config
- `configs/extraction/concept_pooled_v1.yaml`

### Vector config
- `configs/vectors/concept_vectors_v1.yaml`

### New runners
- `generate_concept_corpus.py`
- `collect_concept_pooled_activations.py`
- `extract_concept_vectors.py`

---

## Outputs

### Generated concept corpus
Saved under:
- `outputs/generated/`

Example name pattern:
- `concept_corpus_v1__<model>__c<num_concepts>__g<num_generations>__seed<seed>.jsonl`

### Pooled activations
Saved under:
- `outputs/activations/`

### Concept vectors
Saved under:
- `outputs/vectors/concept_vectors_v1/`

---

## Next steps

The next step should be:
- **controlled sensitivity tests**, not steering

Examples:
- compare texts that differ only in ambiguity type
- compare correct vs generic vs wrong clarification within the same scenario
- check whether each concept vector activates most for its own concept

After that:
- compare concept vectors by layer
- inspect token activations on held-out texts
- only then move to causal intervention / steering

---

### New concept workflow
```bash
python -m asv_ambiguity.runners.generate_concept_corpus \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --data-config configs/data/concept_corpus_v1.yaml
```

```bash
python -m asv_ambiguity.runners.collect_concept_pooled_activations \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --extraction-config configs/extraction/concept_pooled_v1.yaml \
  --dataset outputs/generated/concept_corpus_v1__unsloth_Llama-3.1-8B-Instruct__c4__g1__seed42.jsonl
```

```bash
python -m asv_ambiguity.runners.extract_concept_vectors \
  --vector-config configs/vectors/concept_vectors_v1.yaml \
  --activations outputs/activations/concept_corpus_v1__unsloth_Llama-3.1-8B-Instruct__c4__g1__seed42__unsloth_Llama-3.1-8B-Instruct__concept_pooled.pt
```

In-domain evaluation:
```bash
python -m asv_ambiguity.runners.validate_concept_vectors \
  --activations outputs/activations/concept_corpus_v1__unsloth_Llama-3.1-8B-Instruct__c4__g1__seed42__unsloth_Llama-3.1-8B-Instruct__concept_pooled.pt \
  --vector-dir outputs/vectors/concept_vectors_v1 \
  --layer 12 \
  --split val test \
  --output-dir outputs/concept_eval \
  --top-k 8
```

External corpus sweep:
```bash
python -m asv_ambiguity.runners.sweep_concept_vectors_on_corpus \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --vector-dir outputs/vectors/concept_vectors_v1 \
  --layer 12 \
  --external-jsonl path/to/external_dataset.jsonl \
  --external-text-key text \
  --burn-in-tokens 30 \
  --max-rows 500 \
  --top-k 25 \
  --output-dir outputs/concept_eval_external
```


## Brief record of previous commands used

These are older commands that were used during the earlier binary clarification pipeline and are kept here only as a brief reference.

### Generate seeded dataset
```bash
python -m asv_ambiguity.runners.generate_dataset \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --data-config configs/data/clarification_seeded_v1.yaml
```

### Collect multi-position activations
```bash
python -m asv_ambiguity.runners.collect_activations \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --extraction-config configs/extraction/multi_positions.yaml \
  --dataset outputs/generated/clarification_seeded_v1__unsloth_Llama-3.1-8B-Instruct__gps1__seed42.jsonl
```

### Extract multi-position vectors
```bash
python -m asv_ambiguity.runners.extract_vector \
  --vector-config configs/vectors/multi_position_vectors.yaml \
  --activations outputs/activations/clarification_seeded_v1__unsloth_Llama-3.1-8B-Instruct__gps1__seed42__unsloth_Llama-3.1-8B-Instruct__multi_position.pt
```

### Validate a vector
```bash
python -m asv_ambiguity.runners.validate_vector \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --dataset outputs/generated/clarification_seeded_v1__unsloth_Llama-3.1-8B-Instruct__gps1__seed42.jsonl \
  --vector outputs/vectors/clarification_seeded_v1__last_question_token__layer16.pt \
  --metadata outputs/vectors/clarification_seeded_v1__last_question_token__layer16.json \
  --splits val test
```

### Visualize token activations
```bash
python -u -m asv_ambiguity.runners.visualize_vector_activations \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --vector outputs/vectors/clarification_seeded_v1__last_question_token__layer16.pt \
  --metadata outputs/vectors/clarification_seeded_v1__last_question_token__layer16.json \
  --dataset outputs/generated/clarification_seeded_v1__unsloth_Llama-3.1-8B-Instruct__gps1__seed42.jsonl \
  --num-examples 8 \
  --splits val test \
  --labels positive_response negative_wrong_question \
  --span assistant_only \
  --drop-special-tokens \
  --output-html outputs/visualizations/many_positive_and_wrong.html
```
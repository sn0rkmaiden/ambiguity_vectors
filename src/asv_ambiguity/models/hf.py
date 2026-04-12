from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class HFCausalModel:
    def __init__(self, config: dict[str, Any]):
        model_cfg = config["model"]
        gen_cfg = config.get("generation", {})
        prompt_cfg = config.get("prompting", {})

        dtype_name = model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16"))
        if dtype_name not in _DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

        self.model_name = str(model_cfg["name"])
        self.system_prompt = str(prompt_cfg.get("system_prompt", "You are a careful assistant."))
        self.use_chat_template = bool(model_cfg.get("use_chat_template", True))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "dtype": _DTYPES[dtype_name],
        }
        if model_cfg.get("device_map") is not None:
            model_kwargs["device_map"] = model_cfg["device_map"]
        if model_cfg.get("attn_implementation") is not None:
            model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

        self.generation_kwargs = dict(gen_cfg)
        # Avoid the "max_new_tokens and max_length both set" warning.
        self.generation_kwargs.pop("max_length", None)

        if self.generation_kwargs.get("pad_token_id") is None:
            self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if self.generation_kwargs.get("eos_token_id") is None:
            self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

    def build_messages(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def render_prompt(self, user_prompt: str) -> str:
        messages = self.build_messages(user_prompt)
        if self.use_chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {self.system_prompt}\nUser: {user_prompt}\nAssistant:"

    def generate_text(self, user_prompt: str) -> str:
        prompt_text = self.render_prompt(user_prompt)
        tokenized = self.tokenizer(prompt_text, return_tensors="pt")
        tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            out = self.model.generate(**tokenized, **self.generation_kwargs)

        new_tokens = out[0, tokenized["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def extract_first_question(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^assistant\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()

        for line in cleaned.splitlines():
            line = line.strip()
            if "?" in line:
                q = line[: line.find("?") + 1].strip()
                if q:
                    return q

        match = re.search(r"([^?]*\?)", cleaned)
        if match:
            q = match.group(1).strip()
            if q:
                return q

        raise ValueError(f"Could not extract a question from model output: {text!r}")
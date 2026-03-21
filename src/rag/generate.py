from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class GenerationConfig:
    model: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool = False


class HFGenerator:
    def __init__(self, gen_cfg: GenerationConfig):
        self.cfg = gen_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(gen_cfg.model, use_fast=True)

        # Heuristic: if it's encoder-decoder, use Seq2Seq; else causal LM
        is_seq2seq = False
        try:
            arch = AutoModelForSeq2SeqLM.from_pretrained(gen_cfg.model, torch_dtype="auto", device_map="auto")
            self.model = arch
            is_seq2seq = True
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(gen_cfg.model, torch_dtype="auto", device_map="auto")

        self.is_seq2seq = is_seq2seq

    def build_prompt(self, question: str, context: str) -> str:
        # Simple, stable prompt for QA.
        return (
            "Answer the question using only the provided context. "
            "If the answer cannot be found in the context, say: I don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @torch.inference_mode()
    def generate(self, question: str, context: str) -> str:
        prompt = self.build_prompt(question, context)

        if self.is_seq2seq:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=bool(self.cfg.do_sample),
                temperature=float(self.cfg.temperature),
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # flan-t5 returns only answer content typically; still trim
            return text.strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=bool(self.cfg.do_sample),
            temperature=float(self.cfg.temperature),
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None,
        )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Extract the portion after "Answer:" to make scoring cleaner
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:", 1)[-1]
        return decoded.strip()

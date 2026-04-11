"""Chunk denoising helpers."""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


@dataclass
class DenoiseConfig:
    base_model: str
    adapter_path: str
    batch_size: int = 8
    device: Optional[str] = None  # "cuda", "cpu", or "auto"
    max_new_tokens: int = 256


def _resolve_device(requested: Optional[str]) -> str:
    normalized = (requested or "").strip().lower()
    if normalized in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"Requested denoiser device '{requested}' but CUDA is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return normalized


class ChunkDenoiser:
    def __init__(self, cfg: DenoiseConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)

        self.tok = AutoTokenizer.from_pretrained(cfg.adapter_path)
        base = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model)
        self.model = PeftModel.from_pretrained(base, cfg.adapter_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def denoise_batch(self, texts: List[str]) -> List[str]:
        outs: List[str] = []
        bs = int(self.cfg.batch_size)

        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            prompts = [
                "Fix typos/OCR noise without changing meaning. Preserve numbers and names.\n\n" + t
                for t in batch
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=False,
                temperature=0.0,
            )
            dec = self.tok.batch_decode(gen, skip_special_tokens=True)
            outs.extend([d.strip() for d in dec])

        return outs
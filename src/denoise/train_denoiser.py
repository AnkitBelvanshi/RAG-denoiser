import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import LoraConfig, get_peft_model, TaskType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="JSONL with input_text/target_text")
    ap.add_argument("--out", required=True, help="Output dir for LoRA adapter")
    ap.add_argument("--base_model", default="google/flan-t5-small")
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ds = load_dataset("json", data_files={"train": args.train})["train"]

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # LoRA for T5 attention projections
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],  # good default for T5
    )
    model = get_peft_model(model, lora_cfg)

    def preprocess(ex):
        src = "Fix typos/OCR noise without changing meaning. Preserve numbers and names.\n\n" + ex["input_text"]
        tgt = ex["target_text"]

        x = tok(src, truncation=True, max_length=args.max_source_len)
        y = tok(text_target=tgt, truncation=True, max_length=args.max_target_len)
        x["labels"] = y["input_ids"]
        return x

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    fp16 = torch.cuda.is_available()

    train_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.out, "trainer_out"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
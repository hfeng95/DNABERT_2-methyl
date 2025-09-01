#!/usr/bin/env python3
"""
pretrain.py
Pretrain DNABERT-2 on genomic sequences from a CSV file.
Each row in the CSV should contain a single DNA sequence (e.g., 6400 bp).
"""

import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description="Pretrain DNABERT-2 on genomic sequences")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file containing DNA sequences")
    parser.add_argument("--output_dir", type=str, default="./dnabert2-pretrained", help="Directory to save model")
    parser.add_argument("--pretrained_model", type=str, default="zhihan1996/DNABERT-2-117M",
                        help="DNABERT-2 model checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--max_length", type=int, default=6400, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    # Load CSV
    print(f"Loading sequences from {args.csv_file} ...")
    df = pd.read_csv(args.csv_file, header=None, names=["text"])

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Tokenize sequences
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Load model
    print(f"Loading model {args.pretrained_model} ...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True
    )

    # gradient checkpointing
    model.gradient_checkpointing_enable()
    if hasattr(model.config,'use_cache'):
        model.config.use_cache = False

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=args.lr,
        logging_steps=500,
        report_to="none",  # disable wandb by default
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training ...")
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


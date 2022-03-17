import os
from datetime import datetime, timezone, timedelta
import multiprocessing

from datasets import load_dataset
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
)

IST = timezone(offset=timedelta(hours=5.5))
now = lambda: datetime.now(tz=IST).strftime("%b-%d_%H:%M")
is_distributed = torch.distributed.is_torchelastic_launched

cpu_count = multiprocessing.cpu_count()
max_length = 100


def train(resume=False, distributed=False):
    dataset = load_dataset("diwank/silicone-merged", "balanced")

    labels_set = dataset["train"].unique("labels")
    num_labels = len(labels_set)


    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=num_labels)


    def process(entries):
        return tokenizer(entries["text_a"], entries["text_b"], truncation=True, max_length=max_length)

    tokenized_dataset = dataset \
      .map(process, batched=True) \
      .rename_column("labels", "label") \
      .remove_columns(["text_a", "text_b"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=max_length)


    os.environ["OMP_NUM_THREADS"] = str(cpu_count//2 or 1)
    
    os.environ["WANDB_PROJECT"] = "da-silicone-combined"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_WATCH"] = "all"

    training_args = TrainingArguments(
        weight_decay=0.01,
        learning_rate=5e-5,
        per_device_train_batch_size=320,
        per_device_eval_batch_size=50,
        num_train_epochs=12,
        lr_scheduler_type="cosine",
        warmup_ratio=0.075,

        gradient_accumulation_steps=3,
        fp16=distributed,
        bf16=(not distributed),  # Not compatible with sharded_ddp
        tf32=True,
        group_by_length=True,
        optim="adamw_apex_fused",
        gradient_checkpointing=(not distributed),

        sharded_ddp=distributed and "zero_dp_3 auto_wrap",
        # deepspeed="./ds_config.json",
        do_train=True,
        overwrite_output_dir=(not resume),
        output_dir="./outputs",
        logging_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"deberta_v3-{now()}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

if __name__=="__main__":
    train(distributed=is_distributed())
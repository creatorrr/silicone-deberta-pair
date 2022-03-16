import os
from datetime import datetime, timezone, timedelta

from datasets import load_dataset

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
)

IST = timezone(offset=timedelta(hours=5.5))
now = lambda: datetime.now(tz=IST).strftime("%b-%d_%H:%M")

def train(resume=False):
    dataset = load_dataset("diwank/silicone-merged", "balanced")

    labels_set = dataset["train"].unique("labels")
    num_labels = len(labels_set)


    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=num_labels)


    def process(entries):
        return tokenizer(entries["text_a"], entries["text_b"], truncation=True)

    tokenized_dataset = dataset \
      .map(process, batched=True) \
      .rename_column("labels", "label") \
      .remove_columns(["text_a", "text_b"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    os.environ["WANDB_PROJECT"] = "da-silicone-combined"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_WATCH"] = "all"

    training_args = TrainingArguments(
        weight_decay=0.01,
        learning_rate=3e-5,
        per_device_train_batch_size=100,
        per_device_eval_batch_size=20,
        num_train_epochs=4,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.075,

        fp16=True,
        # sharded_ddp="simple",
        # deepspeed="./ds_config.json",
        do_train=True,
        overwrite_output_dir=(not resume),
        output_dir="./outputs",
        logging_steps=2000,
        save_steps=2000,
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
    train()
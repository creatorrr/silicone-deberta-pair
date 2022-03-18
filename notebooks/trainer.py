import os
from datetime import datetime, timezone, timedelta
import multiprocessing

from datasets import load_dataset
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
)


# Time utils
IST = timezone(offset=timedelta(hours=5.5))
now = lambda: datetime.now(tz=IST).strftime("%b-%d_%H:%M")


# Environment details
is_distributed = torch.distributed.is_torchelastic_launched()
cpu_count = multiprocessing.cpu_count()
gpu_count = torch.cuda.device_count()


# Constants
max_length = 100


# Utils
def setup_environ():

    os.environ["OMP_NUM_THREADS"] = str(cpu_count//(gpu_count if distributed else 1) + 1)
    
    os.environ["WANDB_PROJECT"] = "da-silicone-combined"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_WATCH"] = "all"


def process(tokenizer, entries):
    return tokenizer(
        entries["text_a"],
        entries["text_b"],
        truncation=True,
        max_length=max_length, )


# Core hyperparameters
core_hparams = lambda distributed: dict(
    weight_decay=(0.01 if distributed else 0),  # Set to 0.01 or higher for smaller batch sizes
    learning_rate=5e-5,
    per_device_train_batch_size=320 * (1 if distributed else 7),
    per_device_eval_batch_size=50 * (1 if distributed else 7),
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    warmup_ratio=0.075, )


# Optimizations
optimizations = lambda distributed: dict(
    gradient_accumulation_steps=(4 if distributed else 1),
    fp16=distributed,
    bf16=(not distributed),  # Not compatible with sharded_ddp
    tf32=True,
    group_by_length=True,
    optim="adamw_apex_fused",
    gradient_checkpointing=(not distributed),
    # deepspeed="./ds_config.json",
    sharded_ddp=distributed and "zero_dp_3 auto_wrap", )


# Logging options
logging_options = lambda resume: dict(
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
    run_name=f"deberta_v3-{now()}", )


# Training fn
def train(
        model_name="microsoft/deberta-v3-base", 
        resume=False, 
        distributed=False, 
        use_collator=True ):

    # Declare os env vars, done here so it's executed in each process
    setup_environ()
    
    
    # Load dataset
    dataset = load_dataset("diwank/silicone-merged", "balanced")

    labels_set = dataset["train"].unique("labels")
    num_labels = len(labels_set)


    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, )

    
    # Process dataset
    tokenized_dataset = dataset \
      .map(process, batched=True) \
      .rename_column("labels", "label") \
      .remove_columns(["text_a", "text_b"])

       
    # Create collator for dynamic batching
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        max_length=max_length, 
    ) if use_collator else None


    # Build training arguments
    training_args = TrainingArguments(
        do_train=True,

        **core_hparams(distributed=distributed),
        **optimizations(distributed=distributed),
        **logging_options(resume=resume), )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator, )

    
    # Start training
    trainer.train(resume_from_checkpoint=resume)

    
    return model


if __name__=="__main__":

    setup_environ()
    model = train(distributed=is_distributed)
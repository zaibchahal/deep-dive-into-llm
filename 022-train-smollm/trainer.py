"""
trainer.py
----------
Configures and runs the HF Trainer for causal language model
continued pre-training.

Key choices explained:
  - DataCollatorForLanguageModeling(mlm=False)
      We're training a *causal* (autoregressive) model, not a masked
      language model like BERT.  mlm=False means the collator just
      stacks our pre-built blocks — no masking needed.

  - fp16 / bf16 selection
      Apple Silicon (MPS) doesn't support fp16 training reliably, so
      we detect the device and pick the right precision automatically.

  - gradient_accumulation_steps
      Lets us simulate a larger batch without needing more VRAM.
      effective_batch = per_device_batch * gradient_accumulation_steps
"""

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from config import (
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    NUM_TRAIN_EPOCHS,
    OUTPUT_DIR,
    PER_DEVICE_BATCH_SIZE,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    output_dir: str = OUTPUT_DIR,
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    per_device_batch_size: int = PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    save_steps: int = SAVE_STEPS,
    logging_steps: int = LOGGING_STEPS,
) -> Trainer:
    """
    Build and return a configured HF Trainer.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    train_dataset : Dataset
        Output of tokenizer_utils.tokenize_dataset().
    output_dir : str
        Where checkpoints and the final model are saved.
    num_train_epochs : int
    per_device_batch_size : int
    gradient_accumulation_steps : int
    learning_rate : float
    save_steps : int
    logging_steps : int

    Returns
    -------
    transformers.Trainer
    """
    device = _detect_device()
    print(f"[trainer] device: {device}")

    # On MPS we can't use fp16; bf16 is fine on CUDA Ampere+; CPU uses fp32.
    use_fp16 = device == "cuda"
    use_bf16 = False  # set True if on A100/H100

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM — no masking
    )

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",            # disable wandb / tensorboard by default
        dataloader_num_workers=0,    # 0 avoids fork issues on macOS
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    total_steps = (
        len(train_dataset)
        // (per_device_batch_size * gradient_accumulation_steps)
        * num_train_epochs
    )
    print(f"[trainer] ~{total_steps} optimizer steps planned")
    return trainer

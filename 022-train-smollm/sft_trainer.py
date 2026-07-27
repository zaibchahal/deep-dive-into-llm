"""
sft_trainer.py
--------------
Configures and returns a TRL SFTTrainer for instruction fine-tuning.

Why SFTTrainer instead of the plain Trainer?
  - SFTTrainer (from the TRL library) is purpose-built for SFT.
  - It handles the 'text' column automatically: tokenizes, packs
    examples into blocks, and masks the prompt tokens so loss is only
    computed on the assistant's response — not on the user's question.
  - This is the key difference from CPT: in CPT every token contributes
    to the loss; in SFT only the answer tokens do.

Install TRL if not already installed:
    pip install trl
"""

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from trl import SFTTrainer

from config import (
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SFT_BLOCK_SIZE,
    SFT_NUM_TRAIN_EPOCHS,
    SFT_OUTPUT_DIR,
    SFT_PER_DEVICE_BATCH_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_sft_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    output_dir: str = SFT_OUTPUT_DIR,
    num_train_epochs: int = SFT_NUM_TRAIN_EPOCHS,
    per_device_batch_size: int = SFT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    max_seq_length: int = SFT_BLOCK_SIZE,
) -> SFTTrainer:
    """
    Build and return a configured TRL SFTTrainer.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    train_dataset : Dataset
        Must have a 'text' column (output of sft_dataset.build_sft_dataset).
    output_dir : str
    num_train_epochs : int
    per_device_batch_size : int
    gradient_accumulation_steps : int
    learning_rate : float
    max_seq_length : int
        Maximum token length per example. Examples longer than this are
        truncated; shorter ones are packed together to fill the context.

    Returns
    -------
    trl.SFTTrainer
    """
    device = _detect_device()
    print(f"[sft_trainer] device: {device}")

    use_fp16 = device == "cuda"

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
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        # dataset_text_field tells SFTTrainer which column has the formatted text
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        # packing=True concatenates short examples to fill max_seq_length,
        # maximising GPU utilisation — important when you have many short Q&A pairs
        packing=True,
    )

    total_steps = (
        len(train_dataset)
        // (per_device_batch_size * gradient_accumulation_steps)
        * num_train_epochs
    )
    print(f"[sft_trainer] ~{total_steps} optimizer steps planned")
    return trainer

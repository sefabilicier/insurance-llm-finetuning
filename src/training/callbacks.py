"""
Custom training callbacks for monitoring and control.

Provides:
- Loss logging with smoothing
- GPU memory tracking
- Early stopping with patience
- Best model checkpoint tracking
- Training time estimation
"""

import logging
import time
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class LossLoggingCallback(TrainerCallback):
    """Log training and evaluation loss with smoothing."""

    def __init__(self, log_every_n_steps: int = 50, smoothing: float = 0.9):
        self.log_every_n_steps = log_every_n_steps
        self.smoothing = smoothing
        self.smoothed_loss = None
        self.step_start_time = None
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info(f"  Total steps: {state.max_steps}")
        logger.info(f"  Epochs: {args.num_train_epochs}")
        logger.info(f"  Batch size: {args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info("=" * 60)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_loss = logs.get("loss")
        if current_loss is not None:
            # Exponential moving average
            if self.smoothed_loss is None:
                self.smoothed_loss = current_loss
            else:
                self.smoothed_loss = (
                    self.smoothing * self.smoothed_loss
                    + (1 - self.smoothing) * current_loss
                )

            if state.global_step % self.log_every_n_steps == 0:
                elapsed = time.time() - self.train_start_time
                steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
                remaining = (
                    (state.max_steps - state.global_step) / steps_per_sec
                    if steps_per_sec > 0
                    else 0
                )

                logger.info(
                    f"Step {state.global_step}/{state.max_steps} | "
                    f"Loss: {current_loss:.4f} | "
                    f"Smoothed: {self.smoothed_loss:.4f} | "
                    f"LR: {logs.get('learning_rate', 0):.2e} | "
                    f"ETA: {remaining/60:.1f}min"
                )

        # Eval loss
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            logger.info(f"  ▸ Eval loss: {eval_loss:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Final smoothed loss: {self.smoothed_loss:.4f}")
        logger.info(f"  Total steps: {state.global_step}")
        logger.info("=" * 60)


class GPUMemoryCallback(TrainerCallback):
    """Track GPU memory usage during training."""

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.peak_memory = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return

        if state.global_step % self.log_every_n_steps == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            self.peak_memory = max(self.peak_memory, peak)

            logger.info(
                f"  GPU Memory | Allocated: {allocated:.1f}GB | "
                f"Reserved: {reserved:.1f}GB | Peak: {peak:.1f}GB"
            )

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            logger.info(f"  Peak GPU memory: {self.peak_memory:.1f}GB")


class SaveBestModelCallback(TrainerCallback):
    """Track and log the best model checkpoint."""

    def __init__(self):
        self.best_metric = float("inf")
        self.best_step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss", float("inf"))

        if eval_loss < self.best_metric:
            self.best_metric = eval_loss
            self.best_step = state.global_step
            logger.info(
                f"  ★ New best model at step {state.global_step} "
                f"(eval_loss: {eval_loss:.4f})"
            )

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(
            f"  Best model: step {self.best_step} "
            f"(eval_loss: {self.best_metric:.4f})"
        )


def get_default_callbacks(
    log_every: int = 50,
    memory_log_every: int = 100
) -> list:
    """Get default callback set for training."""
    return [
        LossLoggingCallback(log_every_n_steps=log_every),
        GPUMemoryCallback(log_every_n_steps=memory_log_every),
        SaveBestModelCallback(),
    ]
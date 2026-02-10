"""
Training Module

This module provides training scripts and utilities for vision-language models,
including GRPO (Generalized Reward-based Policy Optimization) training.

Main components:
- grpo_train.py: Main GRPO training script
- trainer/: GRPO trainer implementations
- configs/: DeepSpeed and training configurations
- scripts/: Shell scripts for training
"""

from .trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOConfig

__all__ = [
    'Qwen2VLGRPOTrainer',
    'Qwen2VLGRPOConfig',
]

__version__ = '1.0.0'


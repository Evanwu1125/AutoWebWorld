"""
GRPO Trainer Module

This module provides GRPO (Generalized Reward-based Policy Optimization) 
trainers for vision-language models.
"""

from .grpo_trainer import Qwen2VLGRPOTrainer
from .grpo_config import Qwen2VLGRPOConfig

__all__ = [
    'Qwen2VLGRPOTrainer',
    'Qwen2VLGRPOConfig',
]

__version__ = '1.0.0'


"""
FSM (Finite State Machine) Generation Module

This module provides tools for automatically generating, validating,
and improving Finite State Machines using LLM agents.

Usage:
    from trajectory.fsm.generator import generate_perfect_fsm

    fsm_data = await generate_perfect_fsm(
        theme="E-commerce Platform",
        model="gpt-5",
        target_score=100,
        output_dir="outputs/ecommerce"
    )
"""

from .generator import (
    BaseAgent,
    FSMGeneratorAgent,
    FSMValidatorAgent,
    FSMImproveAgent,
    FSMPerfectGenerator,
    generate_perfect_fsm
)

__all__ = [
    'BaseAgent',
    'FSMGeneratorAgent',
    'FSMValidatorAgent',
    'FSMImproveAgent',
    'FSMPerfectGenerator',
    'generate_perfect_fsm'
]

__version__ = '1.0.0'


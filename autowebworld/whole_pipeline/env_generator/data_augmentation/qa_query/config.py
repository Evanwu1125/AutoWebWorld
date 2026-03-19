"""Configuration for QA Query Generation"""
import os

# API Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# Model Configuration
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_VLM_MODEL = "gpt-4o"

# Generation Parameters
DEFAULT_MAX_QUESTIONS = 9
DEFAULT_CONCURRENCY = 10

# API Pricing (USD per million tokens)
PRICING = {
    "deepseek-chat": {
        "input": 0.21,
        "output": 0.32
    },
    "deepseek-v3.2-exp": {
        "input": 0.25,
        "output": 0.37
    },
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0
    },
    "gemini-2.5-flash": {
        "input": 0.3,
        "output": 2.52
    }
}

# Question Type Distribution
QUESTIONS_PER_TYPE = {
    "single_feature": (2, 3),
    "visual_description": (2, 3),
    "composite_answer": (2, 3)
}


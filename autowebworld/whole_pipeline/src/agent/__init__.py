from .browser import create_image_content_block, load_image_as_base64, screenshot_url
from .config import load_model_config
from .loop import AgentLoop, AgentResult
from .messages import normalize_messages
from .skill import SkillRegistry
from .tools import BUILTIN_TOOLS, VISION_TOOLS, create_builtin_handlers

__all__ = [
    "AgentLoop",
    "AgentResult",
    "BUILTIN_TOOLS",
    "SkillRegistry",
    "VISION_TOOLS",
    "create_builtin_handlers",
    "create_image_content_block",
    "generate_frontend",
    "generate_web",
    "load_image_as_base64",
    "load_model_config",
    "normalize_messages",
    "screenshot_url",
]


def __getattr__(name):
    if name in {"generate_frontend", "generate_web"}:
        from .frontend_codegen import generate_frontend, generate_web

        return {"generate_frontend": generate_frontend, "generate_web": generate_web}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

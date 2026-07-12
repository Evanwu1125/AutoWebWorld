from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Iterable, List, Type, TypeVar

from ..actions.base import ActionHandler, ActionSpec, AsyncActionHandler
from .async_engine import AsyncActionEngine
from .engine import ActionEngine

T = TypeVar("T", bound=object)


def _is_private_module(name: str) -> bool:
    return name.rsplit(".", 1)[-1].startswith("_")


def _walk_modules(package: ModuleType) -> Iterable[ModuleType]:
    if not hasattr(package, "__path__"):
        return []
    modules: List[ModuleType] = []
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if _is_private_module(module_info.name):
            continue
        modules.append(importlib.import_module(module_info.name))
    return modules


def _iter_concrete_subclasses(module: ModuleType, base_cls: Type[T]) -> Iterable[Type[T]]:
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if obj is base_cls or not issubclass(obj, base_cls):
            continue
        if inspect.isabstract(obj):
            continue
        yield obj


def _load_specs() -> Iterable[ActionSpec]:
    from ..actions import spec as spec_pkg

    for module in _walk_modules(spec_pkg):
        for cls in _iter_concrete_subclasses(module, ActionSpec):
            yield cls()


def _load_handlers(handler_base: Type[T]) -> Iterable[T]:
    from ..actions import handler as handler_pkg

    for module in _walk_modules(handler_pkg):
        for cls in _iter_concrete_subclasses(module, handler_base):
            yield cls()


def register_all(engine: ActionEngine | AsyncActionEngine) -> ActionEngine | AsyncActionEngine:
    if isinstance(engine, AsyncActionEngine):
        handler_base: Type[object] = AsyncActionHandler
    else:
        handler_base = ActionHandler
    for spec in _load_specs():
        engine.register_spec(spec)
    for handler in _load_handlers(handler_base):
        engine.register_handler(handler)
    return engine


register_defaults = register_all

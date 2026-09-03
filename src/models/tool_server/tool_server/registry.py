"""
The tool registry.

Adding a tool means adding one directory under ``tools/``. Nothing else
changes: ``service.py`` discovers what is registered and publishes it on
both protocols.

A tool declares itself with the :func:`tool` decorator, so registration
is explicit in the tool's own file rather than inferred from a filename.
The function signature is the tool's API (FastAPI builds the query
parameters and the OpenAPI schema from it, and the MCP tool schema is
derived from that), and the docstring is the model-facing description.

Discovery imports every module under ``tools/``. Import failures are
fatal on purpose: a tool that cannot be imported would otherwise vanish
from the toolbelt silently, and a client would see a shorter tools/list
with no error anywhere.
"""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

TOOLS_PACKAGE = "tool_server.tools"


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool."""

    name: str
    fn: Callable

    @property
    def path(self) -> str:
        """
        REST path. Underscores become hyphens; the MCP tool name stays
        the function name (it is the route's operation_id).
        """
        return f"/v1/tools/{self.name.replace('_', '-')}"

    @property
    def description(self) -> str:
        return (self.fn.__doc__ or "").strip()


_REGISTRY: dict[str, ToolSpec] = {}


def tool(fn: Callable | None = None, *, name: str | None = None) -> Callable:
    """
    Register a function as a tool.

    Usage::

        @tool
        def current_date(timezone: str = "UTC") -> dict:
            \"\"\"Get the current date and time in a timezone.\"\"\"

    ``name`` overrides the tool name, which defaults to the function
    name and is used for the MCP tool name and the REST path.
    """

    def register(f: Callable) -> Callable:
        tool_name = name or f.__name__
        existing = _REGISTRY.get(tool_name)
        if existing is not None and existing.fn is not f:
            raise RuntimeError(
                f"Two tools are registered as {tool_name!r}: "
                f"{existing.fn.__module__} and {f.__module__}. Tool names "
                "must be unique; they are the MCP tool names."
            )
        if not (f.__doc__ or "").strip():
            raise RuntimeError(
                f"Tool {tool_name!r} has no docstring. The docstring is the "
                "description the model reads to decide whether to call it."
            )
        _REGISTRY[tool_name] = ToolSpec(name=tool_name, fn=f)
        return f

    return register(fn) if fn is not None else register


def discover(package: str = TOOLS_PACKAGE) -> list[ToolSpec]:
    """
    Import every module under ``package`` and return the tools found.

    Sorted by name so the published contract, and therefore the tool
    order in every prompt, is deterministic.
    """
    pkg = importlib.import_module(package)

    def on_error(module_name: str) -> None:
        # pkgutil calls this while handling the ImportError, so a bare
        # raise re-raises it. Without this, walk_packages swallows import
        # errors and the tool disappears from the toolbelt silently.
        raise

    for _, module_name, _ in pkgutil.walk_packages(
        pkg.__path__, prefix=f"{pkg.__name__}.", onerror=on_error
    ):
        importlib.import_module(module_name)

    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def registered() -> list[ToolSpec]:
    """Tools registered so far, without triggering discovery."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]

"""
LiftWing tool-server: a toolbelt of universal, read-only tools.

This module publishes whatever ``tools/`` contains; it does not know the
tools by name. Adding a tool means adding one directory under ``tools/``
with an ``@tool``-decorated function, so this file stays the same size as
the toolbelt grows.

One codebase serves two protocols from the same tool functions:

* REST with OpenAPI: what Open WebUI consumes as an external tool
  server, and what humans read at ``/docs``. Each tool becomes
  ``GET /v1/tools/<tool-name>``; the function signature is the query
  contract and the docstring is the description.
* MCP, mounted at ``/mcp``: the standard for tool servers, for MCP
  clients such as an orchestrator. Derived from the REST routes with
  ``FastMCP.from_fastapi``, so a tool's MCP name is its function name
  and its MCP schema comes from the same signature. The exception is a
  tool with an MCP App version (``registry.app_tool``): on MCP that
  version replaces the derived tool under the same name, and serves a
  ``ui://`` View that MCP Apps hosts render with its structured result.

Errors follow the taxonomy in ``errors.py``, mapped to status codes by
the two exception handlers below and nowhere else: ToolInputError is
deterministic (400, do not retry), ToolUpstreamError is transient (502,
may be retried). Tools raise; the service maps.

This service is what clients (who own the loop) call. See T434274#12256418
and T434274#12280495.
"""

import logging
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.apps.config import AppConfig, ResourceCSP
from fastmcp.server.providers.openapi import MCPType, OpenAPITool, RouteMap
from mcp.types import ToolAnnotations

from tool_server.config import LOG_LEVEL
from tool_server.errors import ToolInputError, ToolUpstreamError
from tool_server.registry import AppSpec, ToolSpec, discover, registered_apps

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TOOL_ROUTE_PREFIX = "/v1/tools"

app = FastAPI(
    title="liftwing-tools",
    version="0.1.0",
    description=(
        "A limited toolbelt of universal, read-only tools for LiftWing "
        "clients. REST here, MCP at /mcp. See T436892."
    ),
)


@app.exception_handler(ToolInputError)
async def _tool_input_error(request: Request, exc: ToolInputError) -> JSONResponse:
    """Deterministic tool failure: the client must not retry."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ToolUpstreamError)
async def _tool_upstream_error(
    request: Request, exc: ToolUpstreamError
) -> JSONResponse:
    """Transient tool failure: the client may retry."""
    logger.warning("upstream failure: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Liveness: excluded from the OpenAPI schema and so from MCP."""
    return {"ok": True}


def register_tool_route(fastapi_app: FastAPI, spec: ToolSpec) -> None:
    """
    Publish one tool as a REST route.

    ``operation_id`` is the tool name, which is what MCP uses, and
    ``response_model=None`` keeps FastAPI from re-validating the tool's
    own return type (tools return plain dicts and strings).
    """
    fastapi_app.add_api_route(
        spec.path,
        spec.fn,
        methods=["GET"],
        operation_id=spec.name,
        summary=spec.description.splitlines()[0],
        response_model=None,
        tags=["tools"],
    )


TOOLS = discover()
for _spec in TOOLS:
    register_tool_route(app, _spec)
logger.info("registered %d tools: %s", len(TOOLS), [t.name for t in TOOLS])

# MCP App versions, by the tool they stand in for on MCP.
APPS = {a.tool: a for a in registered_apps()}
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}
_orphans = sorted(set(APPS) - set(_TOOLS_BY_NAME))
if _orphans:
    raise RuntimeError(
        f"MCP App versions of unknown tools: {_orphans}. app_tool(tool=...) "
        "must name a registered @tool."
    )


def annotate(route: object, component: object) -> None:
    """
    Describe how the tools behave, for clients rather than for the model.

    Every tool here reads public data and changes nothing, so they are
    all read-only and idempotent, and they all reach outside this
    service. Clients use these hints to decide whether to ask the user
    before running a tool; annotations cost no prompt tokens, unlike
    saying the same thing in a description.
    """
    if isinstance(component, OpenAPITool):
        component.annotations = tool_annotations(component.name)


def tool_annotations(name: str) -> ToolAnnotations:
    """The annotations every tool gets; see :func:`annotate`."""
    # snake_case field names: the MCP SDK v2 renamed the camelCase ones,
    # which still work but are deprecated.
    return ToolAnnotations(
        title=name.replace("_", " "),
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=True,
    )


def register_app(server: FastMCP, spec: AppSpec, tool_spec: ToolSpec) -> None:
    """
    Publish a tool's MCP App version: the tool, and the View it points at.

    The tool keeps the plain tool's name, description and annotations, so
    the model sees no difference; only MCP Apps hosts act on the UI
    metadata. The View's CSP goes on the resource, where hosts read it
    (LibreChat reads it from the resources/read item, which FastMCP
    stamps from this config). The HTML is read once, here: a missing View
    fails startup rather than the first call.
    """
    html = spec.html.read_text(encoding="utf-8")

    def view() -> str:
        return html

    server.resource(
        spec.resource_uri,
        name=f"{spec.tool}_view",
        description=f"MCP Apps View for {spec.tool}.",
        app=AppConfig(
            csp=ResourceCSP(resource_domains=list(spec.resource_domains) or None)
        ),
    )(view)
    server.tool(
        spec.fn,
        name=spec.tool,
        description=tool_spec.description,
        annotations=tool_annotations(spec.tool),
        app=AppConfig(resource_uri=spec.resource_uri),
    )


# MCP: derived from the routes above, mounted at /mcp.
# The lifespan wiring is required: the MCP session manager initializes in
# the sub-app's lifespan, and FastAPI only runs the lifespan it is given.
mcp = FastMCP.from_fastapi(
    app=app,
    name="liftwing-tools",
    route_maps=[
        # A tool with an MCP App version is not derived: register_app
        # publishes that version under the same name instead. First match
        # wins, so these come before the tool map.
        *(
            RouteMap(
                pattern=rf"^{re.escape(_TOOLS_BY_NAME[name].path)}$",
                mcp_type=MCPType.EXCLUDE,
            )
            for name in APPS
        ),
        # Only the tool routes are tools; nothing else is exposed.
        RouteMap(pattern=rf"^{TOOL_ROUTE_PREFIX}/.*", mcp_type=MCPType.TOOL),
        RouteMap(pattern=r".*", mcp_type=MCPType.EXCLUDE),
    ],
    mcp_component_fn=annotate,
)
for _name, _app_spec in APPS.items():
    register_app(mcp, _app_spec, _TOOLS_BY_NAME[_name])
mcp_app = mcp.http_app(path="/")
app.mount("/mcp", mcp_app)
app.router.lifespan_context = mcp_app.lifespan

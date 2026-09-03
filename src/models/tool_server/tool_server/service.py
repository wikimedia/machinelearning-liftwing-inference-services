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
  and its MCP schema comes from the same signature.

Errors follow the taxonomy in ``errors.py``, mapped to status codes by
the two exception handlers below and nowhere else: ToolInputError is
deterministic (400, do not retry), ToolUpstreamError is transient (502,
may be retried). Tools raise; the service maps.

This service is what clients (who own the loop) call. See T434274#12256418
and T434274#12280495.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.server.openapi import MCPType, RouteMap

from tool_server.config import LOG_LEVEL
from tool_server.errors import ToolInputError, ToolUpstreamError
from tool_server.registry import ToolSpec, discover

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


# MCP: derived from the routes above, mounted at /mcp.
# The lifespan wiring is required: the MCP session manager initializes in
# the sub-app's lifespan, and FastAPI only runs the lifespan it is given.
mcp = FastMCP.from_fastapi(
    app=app,
    name="liftwing-tools",
    route_maps=[
        # Only the tool routes are tools; nothing else is exposed.
        RouteMap(pattern=rf"^{TOOL_ROUTE_PREFIX}/.*", mcp_type=MCPType.TOOL),
        RouteMap(pattern=r".*", mcp_type=MCPType.EXCLUDE),
    ],
)
mcp_app = mcp.http_app(path="/")
app.mount("/mcp", mcp_app)
app.router.lifespan_context = mcp_app.lifespan

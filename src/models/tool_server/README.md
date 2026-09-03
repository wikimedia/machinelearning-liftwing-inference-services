# Tool Server

A limited toolbelt of universal, read-only tools for LiftWing clients.

Since it's an anti-pattern to run tools inside the model-server, clients own the tool-calling loop ([T434274#12256418](https://phabricator.wikimedia.org/T434274#12256418)) and call this service to execute tools ((T434274#12280495)[https://phabricator.wikimedia.org/T434274#12280495]).

This service supports two protocols that serve the same tool functions:

* **REST with OpenAPI**: `GET /v1/tools/<tool>`; live contract at `/openapi.json`, human docs at `/docs`. This is what Open WebUI consumes as an external tool server (no bridge needed).
* **MCP** at `/mcp`: for MCP clients such as an orchestrator. Derived from the REST routes; a route's `operation_id` is the MCP tool name and its docstring is the model-facing description.

There is no static `openapi.yaml`: the contract is generated from the code and served at `/openapi.json`, so it doesn't drift.

## How to run locally

In order to run the tool-server locally, please follow the steps below:

### 1. Build Python venv and install dependencies

First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the server

We can run the server locally with:
```console
uvicorn tool_server.service:app --port 8080
```

### 3. Query the service

Access tools using:

3.1. REST endpoint
```console
curl 'localhost:8080/v1/tools/current-date?timezone=Africa/Kampala'
curl 'localhost:8080/v1/tools/wikipedia-semantic-search?query=CRISPR'
```

3.2. MCP endpoint
```console
python3 - <<'PY'
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8080/mcp/") as c:
        tools = await c.list_tools()
        print("tools/list ->", [t.name for t in tools])
        r = await c.call_tool("current_date", {"timezone": "Africa/Kampala"})
        print("current_date ->", r.data if getattr(r, "data", None) else r.content[0].text)
        r = await c.call_tool("wikipedia_semantic_search",
                              {"query": "CRISPR gene editing", "limit": 2})
        text = r.data if getattr(r, "data", None) else r.content[0].text
        print("wikipedia_semantic_search ->", str(text)[:300], "...")

asyncio.run(main())
PY
```

3.3. TODO: add example that shows full tool-calling loop.


## Tools:

| Tool | What it does | REST | MCP tool name |
|------|--------------|------|---------------|
| `current_date` | Returns the current date, time, and day of the week in an [IANA timezone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) (defaults to `UTC`). | `GET /v1/tools/current-date?timezone=Africa/Kampala` | `current_date` |
| `wikipedia_semantic_search` | Searches Wikipedia for articles matching a natural-language query (semantic search with keyword fallback); returns titles, descriptions, extracts, and source URLs. | `GET /v1/tools/wikipedia-semantic-search?query=CRISPR&language=en&limit=3` | `wikipedia_semantic_search` |

NB: Local runs hit `{language}.wikipedia.org` directly. On LiftWing, set `TOOL_MW_API_PROXY=http://localhost:6500` and requests use the envoy services-proxy with a per-language Host header.

## How to add a new tool

A tool is one `@tool`-decorated function in its own directory under `tools/`. `service.py` discovers whatever `tools/` contains at startup and publishes it on both protocols, so adding a tool never touches `service.py`.

1. Create `tool_server/tools/my_tool/` with an empty `__init__.py` and a `my_tool.py` holding the decorated function:

   ```python
   from typing import Annotated

   from fastapi import Query

   from tool_server.config import DEFAULT_LIMIT, MAX_LIMIT
   from tool_server.errors import ToolInputError, ToolUpstreamError
   from tool_server.registry import tool

   @tool
   def my_tool(
       query: Annotated[str, Query(description="Natural-language query.")],
       limit: Annotated[int, Query(ge=1, le=MAX_LIMIT, description="Max results.")] = DEFAULT_LIMIT,
   ) -> str:
       """Do the thing the tool does.

       This docstring is the model-facing description.
       """
       if not query.strip():
           raise ToolInputError("query must not be empty")
       ...
   ```

   - The function signature is the API: FastAPI builds the query parameters and OpenAPI schema from the `Annotated[..., Query(...)]` annotations, and MCP derives its schema from that.
   - The docstring is the model-facing description; the function name is the tool name (the MCP tool name, and the REST path with underscores turned into hyphens). The `@tool` decorator refuses duplicate names and missing docstrings.
   - Raise `ToolInputError` for deterministic failures and `ToolUpstreamError` for transient upstream failures; the service maps these to `400` and `502`.
   - Read every tunable from `tool_server/config.py`.

2. Add tunables to `tool_server/config.py`. They are env-driven, so operators can override them in Helm values.

3. Add a row to the `## Tools:` table above.

4. Add unit tests in `test/unit/tool_server/test_my_tool.py`:

   ```python
   from tool_server.errors import ToolInputError
   from tool_server.tools.my_tool.my_tool import my_tool
   ```

   Cover the success path and both error paths; mirror the existing tests, which patch the requests session so no network is used.

### Running the unit tests

Install the test and runtime dependencies once, then run pytest with both the repo root and `src/models/tool_server` on `PYTHONPATH`. The latter is what makes `import tool_server` resolve:

```console
pip install -r requirements-test.txt -r src/models/tool_server/requirements.txt
export PYTHONPATH=$PYTHONPATH:.:src/models/tool_server
pytest test/unit/tool_server
```

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
from typing import Any

import httpx
from fastmcp import Client

logger = logging.getLogger("tool_calling_loop")

DEFAULT_TOOLS_URL = "http://localhost:8080/mcp/"
DEFAULT_MODEL_URL = "http://localhost:8585/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen3-0.6b"
DEFAULT_MAX_TURNS = 3
DEFAULT_MAX_TOKENS = 500
REQUEST_TIMEOUT_S = 120.0
# Debian/Ubuntu system trust store, where wmf-certificates installs the
# internal CA. This is the store curl uses.
SYSTEM_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
CA_BUNDLE_ENV_VARS = ("MODEL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one tool-calling loop against the tool-server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", help="The question to ask the model.")
    parser.add_argument(
        "--tools-url",
        default=os.environ.get("TOOLS_URL", DEFAULT_TOOLS_URL),
        help="MCP endpoint of the tool-server.",
    )
    parser.add_argument(
        "--model-url",
        default=os.environ.get("MODEL_URL", DEFAULT_MODEL_URL),
        help="Chat completions endpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="Model name sent in the request body.",
    )
    parser.add_argument(
        "--model-host",
        default=os.environ.get("MODEL_HOST"),
        help=(
            "Host header for the model request, for example "
            "llm-qwen36-27b.llm.wikimedia.org. Required when addressing a "
            "LiftWing inference service by cluster URL."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MAX_TURNS", DEFAULT_MAX_TURNS)),
        help="Loop guard: how many times the model may be asked.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        help="Maximum tokens the model may generate per turn.",
    )
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help=(
            "CA bundle for the model request. Defaults to the first of "
            "MODEL_CA_BUNDLE, REQUESTS_CA_BUNDLE or SSL_CERT_FILE that is "
            f"set, then to {SYSTEM_CA_BUNDLE} when it exists, then to the "
            "certifi bundle httpx ships."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=not env_flag("MODEL_TLS_VERIFY", True),
        help="Skip TLS verification for the model request. Avoid this.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each step of the loop.",
    )
    return parser.parse_args(argv)


def ca_bundle_path(args: argparse.Namespace) -> str | None:
    """
    The CA bundle to trust, or None to use the certifi bundle.

    Prefers an explicit bundle, then the system trust store. The system
    store comes before certifi because the internal CA that signs the
    inference endpoint lives there, which is why curl succeeds where a
    virtualenv's certifi bundle fails.
    """
    if args.ca_bundle:
        return args.ca_bundle
    for name in CA_BUNDLE_ENV_VARS:
        value = os.environ.get(name)
        if value:
            logger.info("using CA bundle from %s: %s", name, value)
            return value
    if os.path.exists(SYSTEM_CA_BUNDLE):
        return SYSTEM_CA_BUNDLE
    return None


def resolve_verify(args: argparse.Namespace) -> ssl.SSLContext | bool:
    """Build what httpx verifies the model's certificate against."""
    if args.insecure:
        return False
    bundle = ca_bundle_path(args)
    if bundle is None:
        return True  # httpx default: the certifi bundle
    if os.path.isdir(bundle):
        return ssl.create_default_context(capath=bundle)
    return ssl.create_default_context(cafile=bundle)


def tool_input_schema(tool: Any) -> dict:
    """
    The tool's JSON schema.

    MCP SDK v2 renamed inputSchema to input_schema; read the new name
    first so the old one is not touched where it is deprecated.
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {"type": "object", "properties": {}}


def build_openai_tools(mcp_tools: list[Any]) -> list[dict]:
    """
    MCP tool definitions -> the OpenAI 'tools' array.

    The mapping is near 1:1: an MCP tool's name, description, and input
    schema are what a chat completions request calls name, description,
    and parameters.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool_input_schema(tool),
            },
        }
        for tool in mcp_tools
    ]


def tool_result_text(result: Any) -> str:
    """Flatten an MCP result into the text of a 'tool' message."""
    data = getattr(result, "data", None)
    if data is not None:
        return json.dumps(data)
    blocks = [getattr(block, "text", "") for block in (result.content or [])]
    return "\n".join(b for b in blocks if b) or "(empty tool result)"


def model_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if args.model_host:
        # The cluster URL is what TLS verifies; the Host header is what the
        # inference gateway routes on. Same shape as the curl calls.
        headers["Host"] = args.model_host
    api_key = os.environ.get("MODEL_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def run_loop(args: argparse.Namespace) -> str:
    """Run the loop and return the model's final answer."""
    if args.insecure:
        logger.warning("TLS verification is disabled for the model request")

    async with (
        Client(args.tools_url) as tools,
        httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_S,
            headers=model_headers(args),
            verify=resolve_verify(args),
        ) as model,
    ):
        # [1] discover the toolbelt, once
        openai_tools = build_openai_tools(await tools.list_tools())
        logger.info(
            "[1] tools/list -> %s", [t["function"]["name"] for t in openai_tools]
        )

        messages: list[dict] = [{"role": "user", "content": args.question}]

        for turn in range(1, args.max_turns + 1):
            # [2] and [4]: ask the model. The call shape is the same both times.
            response = await model.post(
                args.model_url,
                json={
                    "model": args.model,
                    "messages": messages,
                    "tools": openai_tools,
                    "max_tokens": args.max_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            usage = payload.get("usage", {})
            logger.info(
                "[%s] model turn %d: finish_reason=%s prompt_tokens=%s",
                "2" if turn == 1 else "4",
                turn,
                choice["finish_reason"],
                usage.get("prompt_tokens"),
            )

            if choice["finish_reason"] != "tool_calls":
                # [5] the answer is final
                return choice["message"].get("content") or ""

            # Keep the assistant message, with its tool_calls, in the history.
            messages.append(choice["message"])

            for call in choice["message"]["tool_calls"]:
                name = call["function"]["name"]
                # [3] run the tool. The OpenAI format carries arguments as a
                # JSON string and MCP takes a dict, so this parse is the
                # bridge between the two protocols.
                raw_arguments = call["function"]["arguments"] or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"model produced unparseable arguments for {name}: "
                        f"{raw_arguments!r}"
                    ) from e
                logger.info("[3] tools/call %s(%s)", name, arguments)
                try:
                    result = await tools.call_tool(name, arguments)
                    content = tool_result_text(result)
                except Exception as exc:  # surface the failure to the model
                    logger.warning("[3] tools/call %s failed: %s", name, exc)
                    content = f"Tool '{name}' failed: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": content,
                    }
                )

    raise RuntimeError(
        f"no final answer after {args.max_turns} turns; raise --max-turns or "
        "check that the tool results answer the question"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    try:
        print(asyncio.run(run_loop(args)))
    except httpx.HTTPStatusError as e:
        print(
            f"model request failed: HTTP {e.response.status_code} "
            f"{e.response.text[:200]}",
            file=sys.stderr,
        )
        return 1
    except (httpx.HTTPError, RuntimeError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

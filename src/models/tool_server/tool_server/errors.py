"""
Errors shared among the tools.

The taxonomy.

Every tool raises one of these two exceptions, and the service maps them
to HTTP status codes in exactly one place (see the exception handlers in
``service.py``). Tools never choose status codes themselves, so a new
tool inherits the taxonomy instead of restating it.

The distinction is retryability, and it is the client's contract:

* :class:`ToolInputError` is deterministic. The same request will always
  fail the same way, so the client must not retry. The service returns
  400.
* :class:`ToolUpstreamError` is transient. The client may retry. The
  service returns 502.
"""


class ToolInputError(Exception):
    """Deterministic failure: the request itself cannot succeed."""


class ToolUpstreamError(Exception):
    """Transient failure: an upstream dependency was unavailable."""

import asyncio
import csv
import json
import logging
import os
import sys

import aiohttp
import grpc

# grpc_tools.protoc generates flat imports (e.g. "import lambda_pb2") in the
# _grpc.py stub, with no built-in option for relative imports. Adding the proto
# directory to sys.path lets these generated imports resolve without patching
# the stubs after every regeneration.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))

import lambda_pb2
import lambda_pb2_grpc


def load_wiki_languages(path: str) -> dict[str, str]:
    """Load wiki_id -> language_code mapping from TSV."""
    wikis = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                wikis[row[0]] = row[1]
    return wikis


class OutlinkTopicAdapter(lambda_pb2_grpc.LambdaServiceServicer):
    """Adapter between cache service gRPC protocol and OutlinkTopicModel HTTP API."""

    def __init__(self, kserve_url: str, host_header: str, timeout: float = 5):
        self.kserve_url = kserve_url.rstrip("/")
        self.host_header = host_header
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.logger = logging.getLogger(__name__)
        self._session = None

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.wiki_languages = load_wiki_languages(os.path.join(data_dir, "wikis.tsv"))

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a reusable aiohttp session, creating one if needed."""
        if self._session is None or self._session.closed:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "outlink-cache-adapter/1.0 Python/adapter",
            }
            if self.host_header:
                headers["Host"] = self.host_header
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self._session

    async def GetArtifact(
        self, request: lambda_pb2.LambdaRequest, context: grpc.aio.ServicerContext
    ) -> lambda_pb2.LambdaResponse:
        """
        GetArtifact RPC implementation.

        Transforms (wiki, page, revision) -> OutlinkTopicModel inference request.
        When revision is provided, OutlinkTopicModel fetches outlinks for that
        specific revision. Otherwise it uses the current page outlinks.
        """
        self.logger.info(
            f"GetArtifact: wiki={request.wiki} page={request.page} rev={request.revision}"
        )

        try:
            # Map wiki database code to language code using the lookup table
            lang = self.wiki_languages.get(request.wiki)
            if lang is None:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, f"Unknown wiki: {request.wiki}"
                )

            payload = {"page_id": request.page, "lang": lang}
            if request.revision:
                payload["revision_id"] = request.revision

            session = await self._get_session()
            async with session.post(
                self.kserve_url,
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self.logger.error(
                        f"OutlinkTopicModel error {resp.status}: {error_text}"
                    )
                    await context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"OutlinkTopicModel returned {resp.status}",
                    )

                result = await resp.json()

            # Transform OutlinkTopicModel response -> cache service format
            content = json.dumps(result).encode("utf-8")
            metadata = {"Content-Type": "application/json"}

            self.logger.info(f"Success for page {request.page}")
            return lambda_pb2.LambdaResponse(content=content, metadata=metadata)

        except aiohttp.ClientError as e:
            self.logger.error(f"OutlinkTopicModel request failed: {e}")
            await context.abort(
                grpc.StatusCode.UNAVAILABLE, f"OutlinkTopicModel unavailable: {e}"
            )


async def serve():
    """Start the gRPC server."""
    # Configuration
    kserve_url = os.environ.get(
        "KSERVE_URL",
        "http://outlink-topic-model.articletopic-outlink/v1/models/outlink-topic-model:predict",
    )
    host_header = os.environ.get(
        "KSERVE_HOST_HEADER",
        "outlink-topic-model-predictor.articletopic-outlink.svc.cluster.local",
    )
    grpc_port = int(os.environ.get("GRPC_PORT", "50051"))
    timeout = float(os.environ.get("KSERVE_TIMEOUT", "5"))

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Create adapter
    adapter = OutlinkTopicAdapter(
        kserve_url=kserve_url, host_header=host_header, timeout=timeout
    )

    # Start async gRPC server
    server = grpc.aio.server()
    lambda_pb2_grpc.add_LambdaServiceServicer_to_server(adapter, server)
    server.add_insecure_port(f"[::]:{grpc_port}")

    logger.info(f"Starting adapter on port {grpc_port}")
    logger.info(f"Forwarding to: {kserve_url}")
    logger.info(f"Using Host header: {host_header}")

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())

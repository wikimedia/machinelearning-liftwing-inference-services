"""KServe model server for the OpenSearch semantic highlighter.

The OpenSearch neural-search plugin, via an ml-commons remote connector, POSTs a
batch of ``{question, context}`` pairs, and this service returns, per pair, the
offset spans of ``context`` to highlight. It extracts each span with
``timpal0l/mdeberta-v3-base-squad2``, a multilingual SQuAD 2.0 extractive QA
model, so it can also abstain when a passage does not answer the query -- which
yields no highlight rather than a spurious one.

This module is only the KServe shell; the highlighter logic lives in
``highlighter.py``, which imports no serving dependencies. The three KServe
stages delegate to it one for one.

Config: see ``config.py``. Wire contract: see this service's README.
"""

import asyncio
import logging
from typing import Any

import anyio
import kserve
from kserve.errors import InvalidInput

from src.models.semantic_highlighting.model_server.config import settings
from src.models.semantic_highlighting.model_server.highlighter import Highlighter, Pair

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)
logger = logging.getLogger(__name__)


class SemanticHighlighterModel(kserve.Model):
    def __init__(self, name: str, highlighter: Highlighter = None) -> None:
        super().__init__(name)
        self.ready = False
        self.highlighter = highlighter or Highlighter()
        # Serialise inference. A 100-pair batch at 384 tokens through a 1.7 GB
        # fp32 model already fills a GPU; without this, concurrent search
        # requests would run several such batches at once and OOM.
        self._inference_lock = asyncio.Lock()

    def load(self) -> None:
        self.highlighter.load()
        self.ready = True

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> list[Pair]:
        try:
            return self.highlighter.extract_pairs(inputs)
        except ValueError as e:
            raise InvalidInput(f"Wrong request! Message: {e}.")

    async def predict(
        self, pairs: list[Pair], headers: dict[str, str] = None
    ) -> list[list[dict]]:
        # A body carrying no `inputs` extracts no pairs. Answer it straight away
        # rather than queueing behind a 100-pair batch holding the lock.
        if not pairs:
            return []
        # Inference is blocking CPU/GPU work and `predict` is async: run it on a
        # worker thread so a 100-pair batch does not stall the event loop (and
        # with it the readiness and liveness probes).
        async with self._inference_lock:
            spans = await anyio.to_thread.run_sync(self.highlighter.run_batch, pairs)
        return spans

    async def postprocess(
        self, result: list[list[dict]], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        return self.highlighter.envelope(result)


if __name__ == "__main__":
    model = SemanticHighlighterModel(settings.model_name)
    # Load before serving: ModelServer.start() refuses to start unless at least
    # one registered model reports ready.
    model.load()
    # One worker: a single model copy in memory. Scale out with replicas, not
    # with extra in-process workers.
    kserve.ModelServer(workers=1).start([model])

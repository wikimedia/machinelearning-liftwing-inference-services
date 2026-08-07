"""The semantic highlighter itself: request body in, highlight envelope out.

Deliberately knows nothing about KServe. ``model.py`` is a thin ``kserve.Model``
shell that delegates to this class, which keeps the whole wire contract -- pair
extraction, batching, the ``{"highlights": ...}`` envelope -- unit testable
without importing the serving stack or torch.

The stages line up with KServe's:

    extract_pairs   preprocess
    run_batch       predict
    envelope        postprocess
"""

import logging
import time

from pydantic import ValidationError

from python.preprocess_utils import validate_json_input
from src.models.semantic_highlighting.model_server.config import (
    Settings,
    settings as default_settings,
)
from src.models.semantic_highlighting.model_server.qa import (
    answer_question_batch,
    load_model,
)
from src.models.semantic_highlighting.model_server.request_model import BatchRequest
from src.models.semantic_highlighting.model_server.spans import spans_from_result

logger = logging.getLogger(__name__)

# One (question, context) pair.
Pair = tuple[str, str]


class Highlighter:
    def __init__(self, settings: Settings = default_settings) -> None:
        self.settings = settings
        self.tokenizer = None
        self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        logger.info("loading QA model from %s", self.settings.model_path)
        self.tokenizer, self.model = load_model(
            self.settings.model_path, dtype=self.settings.dtype
        )
        logger.info(
            "model loaded on device %s as %s", self.model.device, self.settings.dtype
        )
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one full-size forward pass before the server reports ready.

        The first inference pays one-off costs -- kernel selection and
        autotuning, allocator growth -- that would otherwise land on a real
        request. Those costs are per input *shape*, so this warms the shape the
        service actually runs: ``batch_size`` pairs padded to ``MAX_LENGTH``,
        the widest chunk :meth:`run_batch` can produce.

        Calls ``answer_question_batch`` rather than :meth:`run_batch` on
        purpose. The latter would drop these pairs if their length fell under
        ``min_context_tokens``, and the warm-up would silently do nothing.

        A failure here propagates out of ``load()``, so the model never reports
        ready -- a server that cannot run inference should not take traffic.
        """
        # Deliberately longer than MAX_LENGTH tokens: the tokenizer truncates it
        # to exactly that, making the padded shape the worst case regardless of
        # how this filler happens to tokenize.
        context = "warm up the model " * 200
        pairs = [("what is this", context)] * self.settings.batch_size
        started = time.perf_counter()
        answer_question_batch(
            self.tokenizer,
            self.model,
            pairs,
            max_answer_len=self.settings.max_answer_len,
            top_k=self.settings.top_k,
        )
        logger.info(
            "warm-up pass over %d pairs took %.2fs",
            len(pairs),
            time.perf_counter() - started,
        )

    def extract_pairs(self, body) -> list[Pair]:
        """Validate the request body and pull out the (question, context) pairs.

        Returns the question/context pairs. Raises ``ValueError`` on a body it
        cannot read.

        ``validate_json_input`` decodes the body when KServe hands it over as raw
        bytes, which it does whenever the request's Content-Type is not one of
        its JSON types -- since kserve 0.11 any content type is accepted.
        """
        try:
            request = BatchRequest(**validate_json_input(body))
        except (ValidationError, TypeError) as e:
            raise ValueError(str(e)) from e

        if request.inputs is not None:
            return [(item.question, item.context) for item in request.inputs]
        # Neither shape present. Highlight nothing rather than erroring: a 4xx
        # here fails the caller's entire `_search`, whereas an empty result only
        # leaves the hits unhighlighted.
        return []

    def run_batch(self, pairs: list[Pair]) -> list[list[dict]]:
        """Highlight spans for each pair, preserving order.

        Chunked into batched forward passes rather than one pair at a time, so N
        pairs cost ceil(N / batch_size) inferences. Blocking work: the server
        runs this on a worker thread.

        Passages shorter than ``min_context_tokens`` never reach the model and
        answer with no highlight, which is what an abstention returns anyway.
        They are dropped *before* chunking so the remaining batches close up --
        a short passage otherwise still pads to the longest one beside it.
        """
        # Pre-seeded with the skip result and filled in by index, so a skipped
        # passage keeps its slot -- the response owes one list per input, in
        # request order.
        spans: list[list[dict]] = [[] for _ in pairs]
        todo = list(enumerate(pairs))
        if pairs and self.settings.min_context_tokens > 0:
            # Contexts only, no question and no special tokens: the count has to
            # describe the passage, not the prompt built around it.

            # This "extra" tokenizer pass is cheaper than reusing the encoding
            # `answer_question_batch` builds, because this one asks for nothing
            # but token ids: no padding, no offset mapping, no torch tensors.
            # Tokenizing is only ~15% of that call and materializing its tensors
            # 70-80%, so reusing it would mean building those tensors for the
            # passages that are about to be dropped (and slicing them back
            # down afterward).
            # This is obviously only true if this min_context_tokens optimization
            # is tuned properly. Based on a sample and with min_context_tokens at
            # 40 it drops ~22% of the passages.
            encoded = self.tokenizer(
                [ctx for _, ctx in pairs], add_special_tokens=False
            )["input_ids"]
            todo = [
                (i, pair)
                for (i, pair), ids in zip(todo, encoded)
                if len(ids) >= self.settings.min_context_tokens
            ]
        for start in range(0, len(todo), self.settings.batch_size):
            chunk = todo[start : start + self.settings.batch_size]
            results = answer_question_batch(
                self.tokenizer,
                self.model,
                [pair for _, pair in chunk],
                max_answer_len=self.settings.max_answer_len,
                top_k=self.settings.top_k,
            )
            for (i, (_, ctx)), res in zip(chunk, results):
                spans[i] = spans_from_result(res, ctx, self.settings.min_score)
        return spans

    def envelope(self, spans: list[list[dict]]) -> dict:
        """Wrap the spans in the envelope OpenSearch reads.

        This dict *is* the response body: KServe's v1 REST path returns a custom
        Model's postprocess output verbatim. ml-commons stores it in
        ``dataAsMap`` and the neural-search plugin reads
        ``dataAsMap.get("highlights")``, so anything other than a top-level
        ``highlights`` key silently leaves every hit unhighlighted instead of
        failing.
        """
        return {"highlights": spans}

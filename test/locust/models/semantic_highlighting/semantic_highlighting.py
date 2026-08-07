"""Load test for the semantic highlighter.

Replays real retrieval output rather than synthetic text:
``data/pure_knn_10.json`` holds 600 enwiki queries, each with the text of the 10
best passages a kNN search returned for it. One task sends one query plus its
passages, which is the shape OpenSearch sends per search page -- so batch sizes
and passage lengths are the ones production will actually see. Everything the
load test does not read (ranks, page ids and titles) is stripped from the dump;
see its ``metadata.note``.

    MODEL=semantic_highlighting locust

How many passages ride in one request is the main thing worth sweeping -- it is
the client half of ``SH_BATCH_SIZE``:

    MODEL=semantic_highlighting locust --passages 3

Environment knobs, on top of the shared HOST/NS/MODEL_NAME ones:

    SH_BENCH_PASSAGES  fallback for --passages, for the Makefile target.
    SH_BENCH_SEED      seed the sampling, to compare runs like for like.
"""

import json
import os
import pathlib
import random

from locust import FastHttpUser, constant, events, task

# Relative to this file, not the working directory, so the test also runs from
# the repository root with `locust -f test/locust/locustfile.py`.
DATA_FILE = pathlib.Path(__file__).resolve().parents[2] / "data" / "pure_knn_10.json"

_seed = os.environ.get("SH_BENCH_SEED")
_rng = random.Random(int(_seed) if _seed else None)


def _load_pool():
    """Read the retrieval dump into ``[(query, [passage text, ...]), ...]``."""
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    pool = []
    for entry in raw["query_result_pairs"]:
        query = entry.get("query")
        texts = [text for text in entry.get("passages", []) if text]
        if query and texts:
            pool.append((query, texts))
    if not pool:
        raise RuntimeError(f"no usable query/passage pairs in {DATA_FILE}")
    return pool


POOL = _load_pool()
ALL_PASSAGES = [text for _query, texts in POOL for text in texts]

# Highlight rate is worth watching next to latency: a config that highlights
# nothing is fast for the wrong reason.
_counts = {"passages": 0, "highlighted": 0}


@events.init_command_line_parser.add_listener
def _add_arguments(parser):
    parser.add_argument(
        "--passages",
        type=int,
        default=10,
        env_var="SH_BENCH_PASSAGES",
        help=(
            "Passages per request, taken from the query's own top 10 in rank "
            "order. Above 10, topped up from other queries so you can reach "
            "the 100 OpenSearch may send in one call. Default 10 = one search "
            "page."
        ),
    )


class SemanticHighlighting(FastHttpUser):
    # No think time -- this measures what the service can absorb, not how a
    # human browses. Pace with the user count instead.
    wait_time = constant(0)

    def on_start(self):
        self.passages_per_request = max(1, self.environment.parsed_options.passages)
        self.model_name = os.environ.get("MODEL_NAME", "semantic-highlighter")
        hostname = os.environ.get("HOST", "semantic-highlighter")
        namespace = os.environ.get("NS", "experimental")
        self.endpoint_url = f"/v1/models/{self.model_name}:predict"
        self.headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }

    @task
    def highlight_a_search_page(self):
        wanted = self.passages_per_request
        query, passages = _rng.choice(POOL)
        contexts = list(passages[:wanted])
        while len(contexts) < wanted:
            contexts.append(_rng.choice(ALL_PASSAGES))

        json_body = {"inputs": [{"question": query, "context": c} for c in contexts]}

        with self.client.post(
            self.endpoint_url,
            json=json_body,
            headers=self.headers,
            # Each batch size gets its own row, so one csv keeps a sweep apart.
            # The endpoint stays in the name because the shared locustfile reads
            # the model out of it when it compares against results/.
            name=f"{self.endpoint_url} [{len(contexts)} passages]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:120]}")
                return
            try:
                highlights = response.json()["highlights"]
            except (ValueError, KeyError, TypeError) as exc:
                # A 200 carrying the wrong body leaves every hit unhighlighted
                # in OpenSearch *without* erroring, so it must not be recorded
                # as a success here either.
                response.failure(f"unusable body: {exc}")
                return
            if not isinstance(highlights, list) or len(highlights) != len(contexts):
                got = (
                    len(highlights)
                    if isinstance(highlights, list)
                    else type(highlights).__name__
                )
                response.failure(f"expected {len(contexts)} span lists, got {got}")
                return

            _counts["passages"] += len(highlights)
            _counts["highlighted"] += sum(1 for spans in highlights if spans)
            response.success()


@events.test_start.add_listener
def _announce(environment, **_kwargs):
    print(
        f"[semantic_highlighting] {len(POOL)} queries / {len(ALL_PASSAGES)} "
        f"passages from {DATA_FILE.name}; "
        f"{environment.parsed_options.passages} passages per request"
    )


@events.test_stop.add_listener
def _report(environment, **_kwargs):
    seen = _counts["passages"]
    if not seen:
        return
    hit = _counts["highlighted"]
    print(
        f"[semantic_highlighting] {hit}/{seen} passages returned a highlight "
        f"({hit / seen * 100:.1f}%); the rest were either skipped as too short "
        f"(SH_MIN_CONTEXT_TOKENS) or abstained. Counted per worker process."
    )

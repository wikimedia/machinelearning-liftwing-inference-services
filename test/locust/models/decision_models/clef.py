"""Load test for the Clef decision model server.

The test sends the T440223 example requests to the KServe v1 endpoint.
It rotates through the noul, choice and score questions.

    MODEL=decision_models locust --host https://inference.svc.eqiad.wmnet:30443

The examples are short, about 150 input tokens. To test long inputs, add
fictional wikitext to each state until the request has about N input tokens:

    MODEL=decision_models locust --input-tokens 12000

clef-flash runs only on ml-serve-eqiad, so the staging host in locust.conf
does not work. A local server ignores the Host header, but it uses the
model.py default name:

    MODEL=decision_models MODEL_NAME=clef-flash locust --host http://localhost:8080 -u 8
"""

import itertools
import os

from locust import FastHttpUser, constant, events, task

EDIT = (
    "Edit replaced the article lead with promotional language "
    "and removed three citations."
)

# These are the request bodies from the T440223 examples.
TEXT_BODIES = [
    {
        "state": EDIT,
        "questions": {
            "revert": {"type": "noul", "instructions": "Should this edit be reverted?"}
        },
    },
    {
        "state": (
            "Article about a local restaurant, written by its owner, "
            "with no independent sources."
        ),
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "What should happen to this article?",
                "criteria": {
                    "keep": "Notable enough and fixable by editing.",
                    "improve": "Needs sources and a neutral rewrite before it can stay.",
                    "delete": "Not notable and unlikely to become so.",
                },
            }
        },
    },
    {
        "state": EDIT,
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "How severe is this edit?",
                "criteria": ["Minor", "Moderate", "Severe"],
            }
        },
    },
]

# This is one paragraph of fictional wikitext. Wikitext gives about 3.4
# characters per token, so 12,000 tokens use about 41,000 characters. The
# server refuses a state above MAX_STATE_CHARS (default 60,000).
FILLER_PARAGRAPH = (
    "'''Market Square Kitchen''' is a restaurant in the old town of [[Northbridge]]. "
    "It opened in 1998 in a former bakery on the north side of the [[market square]]."
    "<ref>{{cite news |title=New kitchen for the old bakery |work=Northbridge Gazette "
    "|date=14 March 1998 |page=7}}</ref> The dining room seats about 40 guests, and "
    "the menu follows [[regional cuisine]] with modern touches.<ref>{{cite web "
    "|url=https://example.org/menu |title=Menu |access-date=2 June 2017}}</ref> "
    "The building dates from c. 1885 and was renovated in 2011, when the kitchen "
    "moved to the ground floor.\n\n== History ==\n"
    "The owner held cooking classes from 2003 to 2009, a monthly wine evening, and a "
    "summer festival ({{circa|2012}}–2019) for the neighbourhood."
    '<ref name="gazette2017"/>'
)
# The clef-flash tokenizer gives 222 tokens for one paragraph and its separator.
FILLER_PARAGRAPH_TOKENS = 222
# The prompt template, the questions and the short state give about 150 tokens.
BASE_TOKENS = 150

# The server returns the real token count. The test reports it at the end.
_input_tokens = {"requests": 0, "total": 0, "min": None, "max": None}


@events.init_command_line_parser.add_listener
def _add_arguments(parser):
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=0,
        env_var="CLEF_BENCH_INPUT_TOKENS",
        help=(
            "Approximate input tokens per request. The test adds fictional "
            "wikitext to each state. The server truncates above 16,384 "
            "tokens. Default 0 sends the short examples."
        ),
    )


def pad_body(body: dict, input_tokens: int) -> dict:
    """Add filler paragraphs to the state until the body has about input_tokens."""
    paragraphs = max(1, round((input_tokens - BASE_TOKENS) / FILLER_PARAGRAPH_TOKENS))
    filler = "\n\n".join([FILLER_PARAGRAPH] * paragraphs)
    return {**body, "state": f"{body['state']}\n\nArticle wikitext:\n\n{filler}"}


def record_input_tokens(tokens: int) -> None:
    stats = _input_tokens
    stats["requests"] += 1
    stats["total"] += tokens
    stats["min"] = tokens if stats["min"] is None else min(stats["min"], tokens)
    stats["max"] = tokens if stats["max"] is None else max(stats["max"], tokens)


class ClefFlash(FastHttpUser):
    # The test measures the capacity of the service, so users do not wait.
    # Set the load with the user count.
    wait_time = constant(0)
    # The transformers server runs one request at a time. With many users and
    # long inputs, a request can wait in the queue for longer than the
    # default 60 s timeout.
    network_timeout = 600.0

    def on_start(self):
        self.model_name = os.environ.get("MODEL_NAME", "llm-clef-flash")
        hostname = os.environ.get("HOST", "llm-clef-flash")
        namespace = os.environ.get("NS", "llm")
        self.endpoint_url = f"/v1/models/{self.model_name}:predict"
        self.headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }
        input_tokens = self.environment.parsed_options.input_tokens
        bodies = TEXT_BODIES
        # The short examples keep the plain endpoint as the request name, so
        # they match results/clef-flash_stats.csv. Long inputs get their own row.
        self.request_name = self.endpoint_url
        if input_tokens > 0:
            bodies = [pad_body(body, input_tokens) for body in TEXT_BODIES]
            self.request_name = f"{self.endpoint_url} [{input_tokens} tokens]"
        self.bodies = itertools.cycle(bodies)

    @task
    def get_prediction(self):
        with self.client.post(
            self.endpoint_url,
            json=next(self.bodies),
            headers=self.headers,
            name=self.request_name,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:120]}")
                return
            try:
                result = response.json()
                answers = result["answers"]
            except (ValueError, KeyError, TypeError) as e:
                response.failure(f"unusable body: {e!r}")
                return
            if not answers:
                response.failure("empty answers")
                return
            response.success()
            tokens = (result.get("usage") or {}).get("input_tokens")
            if isinstance(tokens, int):
                record_input_tokens(tokens)


@events.test_stop.add_listener
def _report(environment, **_kwargs):
    stats = _input_tokens
    if not stats["requests"]:
        return
    print(
        f"[clef] input_tokens per request: mean "
        f"{stats['total'] / stats['requests']:.0f} (min {stats['min']}, "
        f"max {stats['max']}) over {stats['requests']} requests. "
        "Counted per worker process."
    )

"""Load test for the cope-b-a4b policy violation model server.

The test sends the 100 synthetic requests from data/cope_b_synthetic.json to
the KServe v1 :predict endpoint. Each prompt has about 2,500 tokens, and the
model gives 1 output token. This is the shape of the production traffic
(T440311). To make the data again, run data/cope_b_synthetic.py.

By default, the test sends requests to the copy in the experimental namespace,
not to the production service in the llm namespace:

    MODEL=cope_b locust --host https://inference.svc.eqiad.wmnet:30443 -u 8

To send requests to the production service, set NS=llm. A local server ignores
the Host header:

    MODEL=cope_b locust --host http://localhost:8080 -u 1

At the end of the run, the test prints the two SLIs for the requests that it
sent: availability (non-5xx responses) and latency (2xx responses in 2000 ms
or less). Use --slo-latency-ms to change the latency threshold.
"""

import json
import os
import random

from locust import FastHttpUser, constant, events, task

DATA_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cope_b_synthetic.json"
)

# The SLI counters for the requests of this test. The counters are per
# worker process.
_sli = {"total": 0, "5xx": 0, "2xx": 0, "2xx_fast": 0, "prompt_tokens": 0}


@events.init_command_line_parser.add_listener
def _add_arguments(parser):
    parser.add_argument(
        "--slo-latency-ms",
        type=int,
        default=2000,
        env_var="COPE_B_SLO_LATENCY_MS",
        help="Latency SLI threshold in ms. Default 2000.",
    )


def load_requests() -> list[tuple[dict, int]]:
    """Return the request bodies and their prompt tokens from the data file."""
    with open(DATA_FILE) as f:
        data = json.load(f)
    policies = data["policies"]
    return [
        (
            {"policy": policies[s["policy_id"]], "content": s["content"]},
            s["prompt_tokens"],
        )
        for s in data["samples"]
    ]


class CopeB(FastHttpUser):
    # The test measures the capacity of the service, so users do not wait.
    # Set the load with the user count.
    wait_time = constant(0)
    network_timeout = 120.0

    def on_start(self):
        self.model_name = os.environ.get("MODEL_NAME", "cope-b-a4b")
        hostname = os.environ.get("HOST", "cope-b-a4b")
        namespace = os.environ.get("NS", "experimental")
        self.endpoint_url = f"/v1/models/{self.model_name}:predict"
        self.headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }
        self.requests = load_requests()

    @task
    def get_prediction(self):
        body, prompt_tokens = random.choice(self.requests)
        # vLLM caches the prompt prefixes. The 100 prompts repeat, so without
        # a change the cache keeps each full prompt and the latency is too
        # low. A unique first line in the content stops this. The policy part
        # of the prompt stays the same, so the cache can keep it, as in
        # production.
        content = f"Revision ID: {random.randint(10**8, 10**9 - 1)}\n\n"
        body = {**body, "content": content + body["content"]}
        with self.client.post(
            self.endpoint_url,
            json=body,
            headers=self.headers,
            name=self.endpoint_url,
            catch_response=True,
        ) as response:
            record_sli(
                response.status_code,
                response.request_meta["response_time"],
                prompt_tokens,
                self.environment.parsed_options.slo_latency_ms,
            )
            if response.status_code != 200:
                # The text is None after a connection error.
                text = (response.text or "")[:120]
                response.failure(f"HTTP {response.status_code}: {text}")
                return
            try:
                violation = response.json()["violation"]
            except (ValueError, KeyError, TypeError) as e:
                response.failure(f"unusable body: {e!r}")
                return
            if violation not in (0, 1):
                response.failure(f"unexpected violation value: {violation!r}")
                return
            response.success()


def record_sli(
    status_code: int, response_time: float, prompt_tokens: int, threshold_ms: int
) -> None:
    # Locust gives status 0 for a connection error or a timeout. The gateway
    # does not see these requests, so they are not part of the SLIs. They
    # show as failures in the locust stats.
    if status_code == 0:
        return
    _sli["total"] += 1
    _sli["prompt_tokens"] += prompt_tokens
    if status_code >= 500:
        _sli["5xx"] += 1
    elif 200 <= status_code < 300:
        _sli["2xx"] += 1
        if response_time <= threshold_ms:
            _sli["2xx_fast"] += 1


@events.test_stop.add_listener
def _report(environment, **_kwargs):
    if not _sli["total"]:
        return
    threshold_ms = environment.parsed_options.slo_latency_ms
    availability = 100 * (1 - _sli["5xx"] / _sli["total"])
    latency = 100 * _sli["2xx_fast"] / _sli["2xx"] if _sli["2xx"] else 0.0
    print(
        f"[cope-b] availability SLI: {availability:.2f}% non-5xx "
        f"({_sli['5xx']} 5xx of {_sli['total']} responses). "
        f"Latency SLI: {latency:.2f}% of {_sli['2xx']} 2xx responses in "
        f"{threshold_ms} ms or less. Mean prompt tokens "
        f"{_sli['prompt_tokens'] / _sli['total']:.0f} (from the data file). "
        "Counted per worker process."
    )

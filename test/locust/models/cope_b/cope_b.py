"""Load test for the cope-b-a4b policy violation model server.

The test sends the 100 synthetic requests from data/cope_b_synthetic.json to
the KServe v1 :predict endpoint. Each prompt has about 2,500 tokens, and the
model gives 1 output token. This is the shape of the production traffic
(T440311).

By default, the test sends requests to the copy in the experimental namespace,
not to the production service in the llm namespace:

    MODEL=cope_b locust --host https://inference.svc.eqiad.wmnet:30443 -u 8

To send requests to the production service, set NS=llm. A local server ignores
the Host header:

    MODEL=cope_b locust --host http://localhost:8080 -u 1
"""

import json
import os
import random

from locust import FastHttpUser, constant, task

DATA_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cope_b_synthetic.json"
)


def load_requests() -> list[dict]:
    """Return the request bodies from the data file."""
    with open(DATA_FILE) as f:
        data = json.load(f)
    policies = data["policies"]
    return [
        {"policy": policies[s["policy_id"]], "content": s["content"]}
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
        body = random.choice(self.requests)
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

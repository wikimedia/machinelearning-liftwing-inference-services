import random
import os

from locust import FastHttpUser, between, task


def get_random_input_params():
    max_tokens = random.randint(10, 1000)
    num_words = random.randint(5, 200)
    prompt = " ".join(["What is Wikipedia"] * num_words)
    return prompt, max_tokens


class HuggingfaceServer(FastHttpUser):
    wait_time = between(1, 5)

    @task(3)
    def get_prediction(self):
        prompt, max_tokens = get_random_input_params()
        model = os.environ.get("MODEL_NAME", None)
        hostname = os.environ.get("HOST", None)
        namespace = os.environ.get("NS", "experimental")
        headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }
        self.client.post(
            "/openai/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
            headers=headers,
        )

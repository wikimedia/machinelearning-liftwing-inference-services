import random
import os

from locust import FastHttpUser, between, task


def get_random_input_params():
    num_words = random.randint(5, 200)
    original = " ".join(["What is Wikipedia"] * num_words)
    modified = " ".join(["What is the amazing Wikipedia site"] * num_words)
    return original, modified


class EditCheckPeacock(FastHttpUser):
    wait_time = between(0.0, 0.1)

    @task
    def get_prediction(self):
        original, modified = get_random_input_params()
        hostname = os.environ.get("HOST", "edit-check")
        namespace = os.environ.get("NS", "experimental")
        headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }
        self.client.post(
            "/v1/models/edit-check-staging:predict",
            json={
                "lang": "en",
                "check_type": "peacock",
                "original_text": original,
                "modified_text": modified,
            },
            headers=headers,
        )

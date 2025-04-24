import random
import os
import json

from locust import FastHttpUser, between, task


def get_random_input_params():
    num_words = random.randint(5, 20)
    original = " ".join(["What is Wikipedia"] * num_words)
    modified = " ".join(["What is the amazing Wikipedia site"] * num_words)
    return original, modified


def get_random_batch(n_instances: int = 1):
    batch_list = []
    for _ in range(n_instances):
        original, modified = get_random_input_params()
        dd = {
            "lang": "en",
            "check_type": "peacock",
            "original_text": original,
            "modified_text": modified,
        }
        batch_list.append(dd)
    return {"instances": batch_list}


class EditCheckPeacock(FastHttpUser):
    wait_time = between(0.0, 0.1)

    @task
    def get_prediction(self):
        json_body = get_random_batch(n_instances=1)
        hostname = os.environ.get("HOST", "edit-check")
        namespace = os.environ.get("NS", "experimental")
        endpoint_url = "/v1/models/edit-check-staging:predict"
        headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }
        with self.client.post(
            endpoint_url,
            json=json_body,
            headers=headers,
            catch_response=True,
        ) as response:
            try:
                resp = response.json()
                for pred in resp["predictions"]:
                    status_code = pred["status_code"]
                    if status_code != 200:
                        response.failure(f"{pred['errors']} status_code: {status_code}")
            except (KeyError, json.decoder.JSONDecodeError):
                pass

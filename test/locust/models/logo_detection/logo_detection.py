import os
import base64
import random

from locust import FastHttpUser, between, task

def get_random_image(image_dir):
    image_files = os.listdir(image_dir)
    filename = random.choice(image_files)
    image_path = os.path.join(image_dir, filename)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return filename, encoded_image

class LogoDetection(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        filename, encoded_image = get_random_image(
            "../../src/models/logo_detection/data/"
        )
        headers = {
            "Content-Type": "application/json",
            "Host": "logo-detection.experimental.wikimedia.org",
        }
        self.client.post(
            "/v1/models/logo-detection:predict",
            json={
                "instances": [
                    {"filename": filename, "image": encoded_image, "target": "logo"}
                ],
                "debug": True,
            },
            headers=headers,
        )
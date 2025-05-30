import random

from locust import FastHttpUser, task


def _get_random_text() -> str:
    num_words = random.randint(5, 100)
    return " ".join(["word" for _ in range(num_words)])


class LanguageIdentificationModel(FastHttpUser):
    @task
    def get_prediction(self) -> None:
        headers = {
            "Content-Type": "application/json",
            "Host": "langid.llm.wikimedia.org",
        }
        self.client.post(
            "/v1/models/langid:predict",
            json={"text": _get_random_text()},
            headers=headers,
        )

import pandas as pd

from locust import FastHttpUser, between, task

pages = pd.read_csv("data/editing_suggestions_locust.csv")


def get_random_sample_from_df_input(df):
    page_id, wiki_id = df.sample(n=1).squeeze().tolist()
    return page_id, wiki_id


class EditingSuggestions(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        page_id, wiki_id = get_random_sample_from_df_input(pages)
        headers = {
            "Content-Type": "application/json",
            "Host": "editing-suggestions.experimental.wikimedia.org",
        }
        self.client.post(
            "/v1/models/editing-suggestions:predict",
            json={"wiki_id": wiki_id, "page_id": int(page_id)},
            headers=headers,
        )

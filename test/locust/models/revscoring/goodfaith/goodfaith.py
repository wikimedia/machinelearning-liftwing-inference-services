import pandas as pd
from locust import FastHttpUser, task, between

rev_ids = pd.read_csv("inputs/enwiki_revids.input", delimiter="\t", header=None)


def get_random_sample_from_df_input(df):
    rev_id = df.sample(n=1).squeeze().tolist()
    return rev_id


class EnwikiGoodfaith(FastHttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.client.headers = {"Content-Type": "application/json"}

    @task
    def get_prediction(self):
        rev_id = get_random_sample_from_df_input(rev_ids)
        self.client.post(
            "/service/lw/inference/v1/models/enwiki-goodfaith:predict",
            json={"rev_id": int(rev_id)},
        )

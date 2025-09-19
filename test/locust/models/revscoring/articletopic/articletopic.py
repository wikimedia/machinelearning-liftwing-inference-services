import pandas as pd

from locust import FastHttpUser, between, task

rev_ids = pd.read_csv("data/enwiki_revids.tsv", delimiter="\t", header=None)


def get_random_sample_from_df_input(df):
    rev_id = df.sample(n=1).squeeze().tolist()
    return rev_id


class EnwikiArticletopic(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        headers = {
            "Content-Type": "application/json",
            "Host": "enwiki-articletopic.revscoring-articletopic.wikimedia.org",
        }
        rev_id = get_random_sample_from_df_input(rev_ids)
        self.client.post(
            "/v1/models/enwiki-articletopic:predict",
            json={"rev_id": int(rev_id)},
            headers=headers,
        )

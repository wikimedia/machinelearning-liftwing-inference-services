import pandas as pd

from locust import FastHttpUser, between, task

rev_ids = pd.read_csv("data/wikidata_revids.tsv", delimiter="\t", header=None)


def get_random_sample_from_df_input(df):
    rev_id = df.sample(n=1).squeeze().tolist()
    return rev_id


class RevertriskWikidata(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        rev_id = get_random_sample_from_df_input(rev_ids)
        headers = {
            "Content-Type": "application/json",
            "Host": "revertrisk-wikidata.revision-models.wikimedia.org",
        }
        self.client.post(
            "/v1/models/revertrisk-wikidata:predict",
            json={
                "rev_id": rev_id,
            },
            headers=headers,
        )

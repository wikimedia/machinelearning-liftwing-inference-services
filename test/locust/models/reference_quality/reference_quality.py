import pandas as pd

from locust import FastHttpUser, between, task

rev_ids = pd.read_csv("inputs/sample_all.input", delimiter="\t", header=None)


def get_random_sample_from_df_input(df, skip_wikis=[]):
    lang, rev_id = df[~df[0].isin(skip_wikis)].sample(n=1).squeeze().tolist()
    return lang, rev_id


class ReferenceNeed(FastHttpUser):
    wait_time = between(1, 5)

    @task(3)
    def get_prediction(self):
        lang, rev_id = get_random_sample_from_df_input(rev_ids)
        headers = {
            "Content-Type": "application/json",
            "Host": "reference-quality.experimental.wikimedia.org",
        }
        self.client.post(
            "/v1/models/reference-need:predict",
            json={"rev_id": int(rev_id), "lang": lang},
            headers=headers,
        )


class ReferenceRisk(FastHttpUser):
    wait_time = between(1, 5)

    @task(3)
    def get_prediction(self):
        # ref-risk model currently does not support de and es wikis
        lang, rev_id = get_random_sample_from_df_input(rev_ids, skip_wikis=["de", "es"])
        headers = {
            "Content-Type": "application/json",
            "Host": "reference-quality.experimental.wikimedia.org",
        }
        self.client.post(
            "/v1/models/reference-risk:predict",
            json={"rev_id": int(rev_id), "lang": lang},
            headers=headers,
        )

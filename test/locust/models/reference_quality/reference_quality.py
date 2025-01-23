import pandas as pd

from locust import FastHttpUser, between, task

input_df = pd.read_csv("data/recentchanges_en.tsv", delimiter="\t")


def get_random_sample_from_df_input(df, skip_wikis=[]):
    return df[~df["lang"].isin(skip_wikis)].sample(n=1).squeeze().tolist()


class ReferenceNeed(FastHttpUser):
    wait_time = between(1, 5)

    @task(3)
    def get_prediction(self):
        sample = get_random_sample_from_df_input(input_df)
        headers = {
            "Content-Type": "application/json",
            "Host": "reference-quality.revision-models.wikimedia.org",
        }
        self.client.post(
            "/v1/models/reference-need:predict",
            json={"rev_id": int(sample[1]), "lang": sample[0]},
            headers=headers,
        )


class ReferenceRisk(FastHttpUser):
    wait_time = between(1, 5)

    @task(3)
    def get_prediction(self):
        # ref-risk model currently does not support de and es wikis
        sample = get_random_sample_from_df_input(input_df, skip_wikis=["de", "es"])
        headers = {
            "Content-Type": "application/json",
            "Host": "reference-quality.revision-models.wikimedia.org",
        }
        self.client.post(
            "/v1/models/reference-risk:predict",
            json={"rev_id": int(sample[1]), "lang": sample[0]},
            headers=headers,
        )

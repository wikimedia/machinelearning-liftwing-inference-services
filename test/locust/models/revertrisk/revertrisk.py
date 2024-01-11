import pandas as pd
from locust import FastHttpUser, task, between

rev_ids = pd.read_csv("inputs/revertrisk.input", delimiter="\t", header=None)


def get_random_sample_from_df_input(df):
    lang, rev_id = df.sample(n=1).squeeze().tolist()
    return lang, rev_id


class RevertriskLanguageAgnostic(FastHttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.client.headers = {"Content-Type": "application/json"}

    @task(3)
    def get_prediction(self):
        lang, rev_id = get_random_sample_from_df_input(rev_ids)
        self.client.post(
            "/service/lw/inference/v1/models/revertrisk-language-agnostic:predict",
            json={"rev_id": int(rev_id), "lang": lang},
        )


class RevertriskMultilingual(FastHttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.client.headers = {"Content-Type": "application/json"}

    @task(3)
    def get_prediction(self):
        lang, rev_id = get_random_sample_from_df_input(rev_ids)
        self.client.post(
            "/service/lw/inference/v1/models/revertrisk-multilingual:predict",
            json={"rev_id": int(rev_id), "lang": lang},
        )

import pandas as pd

from locust import FastHttpUser, between, task

revisions = pd.read_csv("data/revisions_lang_and_id.tsv", delimiter="\t", header=None)


def get_random_sample_from_df_input(df):
    lang, rev_id = df.sample(n=1, random_state=42).squeeze().tolist()
    return lang, rev_id


class ArticlequalityLanguageAgnostic(FastHttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def get_prediction(self):
        lang, rev_id = get_random_sample_from_df_input(revisions)
        headers = {
            "Content-Type": "application/json",
            "Host": "articlequality.article-models.wikimedia.org",
        }
        self.client.post(
            "/v1/models/articlequality:predict",
            json={"rev_id": int(rev_id), "lang": lang},
            headers=headers,
        )

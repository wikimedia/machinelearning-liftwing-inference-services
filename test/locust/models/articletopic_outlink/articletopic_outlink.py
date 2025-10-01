import pandas as pd

from locust import FastHttpUser, between, task

articles = pd.read_csv("data/articles_lang_title_id.csv")


def get_random_sample_from_df_input(df):
    lang, page_title, page_id = df.sample(n=1).squeeze().tolist()
    return lang, page_title, page_id


class ArticleTopicOutlink(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction_title(self):
        lang, page_title, page_id = get_random_sample_from_df_input(articles)
        headers = {
            "Content-Type": "application/json",
            "Host": "outlink-topic-model.articletopic-outlink.wikimedia.org",
        }
        self.client.post(
            "/v1/models/outlink-topic-model:predict",
            json={
                "lang": lang,
                "page_title": page_title,
            },
            headers=headers,
        )

    @task
    def get_prediction_id(self):
        lang, page_title, page_id = get_random_sample_from_df_input(articles)
        headers = {
            "Content-Type": "application/json",
            "Host": "outlink-topic-model.articletopic-outlink.wikimedia.org",
        }
        self.client.post(
            "/v1/models/outlink-topic-model:predict",
            json={
                "lang": lang,
                "page_id": int(page_id),
            },
            headers=headers,
        )

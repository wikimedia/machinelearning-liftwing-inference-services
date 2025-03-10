import pandas as pd

from locust import FastHttpUser, between, task

articles = pd.read_csv("data/articles_lang_and_title.csv")


def get_random_sample_from_df_input(df):
    lang, title = df.sample(n=1).squeeze().tolist()
    return lang, title


class ArticleTopicOutlink(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        lang, title = get_random_sample_from_df_input(articles)
        headers = {
            "Content-Type": "application/json",
            "Host": "outlink-topic-model.articletopic-outlink.wikimedia.org",
        }
        self.client.post(
            "/v1/models/outlink-topic-model:predict",
            json={
                "lang": lang,
                "page_title": title,
            },
            headers=headers,
        )

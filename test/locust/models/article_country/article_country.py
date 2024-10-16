import pandas as pd

from locust import FastHttpUser, between, task

articles = pd.read_csv("inputs/article_country.csv")


def get_random_sample_from_df_input(df):
    lang, title = df.sample(n=1).squeeze().tolist()
    return lang, title


class ArticleCountry(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        lang, title = get_random_sample_from_df_input(articles)
        headers = {
            "Content-Type": "application/json",
            "Host": "article-country.experimental.wikimedia.org",
        }
        self.client.post(
            "/v1/models/article-country:predict",
            json={
                "lang": lang,
                "title": title,
            },
            headers=headers,
        )

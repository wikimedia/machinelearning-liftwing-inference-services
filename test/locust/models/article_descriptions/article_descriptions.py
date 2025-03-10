import pandas as pd

from locust import FastHttpUser, between, task

articles = pd.read_csv("data/articles_lang_and_title.csv")


def get_random_sample_from_df_input(df):
    lang, title = df.sample(n=1).squeeze().tolist()
    return lang, title


class ArticleDescriptions(FastHttpUser):
    wait_time = between(1, 5)

    @task
    def get_prediction(self):
        lang, title = get_random_sample_from_df_input(articles)
        headers = {
            "Content-Type": "application/json",
            "Host": "article-descriptions.experimental.wikimedia.org",
        }
        self.client.post(
            "/v1/models/article-descriptions:predict",
            json={
                "lang": lang,
                "title": title,
                "num_beams": 3,
            },
            headers=headers,
        )


class ArticleDescriptionsCloudVPS(FastHttpUser):
    """
    This class is used to test the cloud VPS version of the article descriptions model.
    It is not enabled by default as it uses a different host
    (https://ml-article-description-api.wmcloud.org)
    """

    wait_time = between(1, 5)

    @task
    def get_prediction_cloud_vps(self):
        lang, title = get_random_sample_from_df_input(articles)
        params = {"lang": lang, "title": title, "num_beams": 3}
        self.client.get("/article", params=params)

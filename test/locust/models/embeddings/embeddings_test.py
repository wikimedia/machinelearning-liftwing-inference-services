import os

import pandas as pd

from locust import FastHttpUser, between, task

min_length = int(os.getenv("MIN_LENGTH", 10))
max_length = int(os.getenv("MAX_LENGTH", 350))
# min_length=10, max_length=350: median = 73
print(f"Min length: {min_length}, Max length: {max_length}")
df = pd.read_csv("data/embedding_questions.csv")
df["question_length"] = df["question"].apply(lambda x: len(x))
df = df[df["question_length"] >= min_length]
df = df[df["question_length"] <= max_length]
print(df[["question_length"]].describe())


def get_prompt(question: str) -> str:
    return f"""Instruct: Given a web search query, retrieve relevant passages that answer the query
                Query: {question}"""


print("Prompt length: ", len(get_prompt("")))


class Embeddings(FastHttpUser):
    wait_time = between(0.0, 0.1)

    @task
    def get_prediction(self):
        hostname = os.environ.get("HOST", "embeddings")
        namespace = os.environ.get("NS", "llm")
        headers = {
            "Content-Type": "application/json",
            "Host": f"{hostname}.{namespace}.wikimedia.org",
        }
        questions = list(df.sample(n=1)["question"])
        prompts = [get_prompt(question) for question in questions]
        self.client.post(
            "/v1/models/qwen3-embedding:predict",
            json={"input": prompts},
            headers=headers,
        )

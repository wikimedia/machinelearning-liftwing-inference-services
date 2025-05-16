from pydantic import BaseModel


class ArticleInstance(BaseModel):
    rev_id: int
    lang: str


class RequestModel(BaseModel):
    instances: list[ArticleInstance]

from pydantic_settings import BaseSettings


"""
Config class for article quality project.
Reads variables as environment variables.
If environment variables do not exist, it uses the default values.
"""


class Settings(BaseSettings):
    model_name: str = "articlequality"
    model_path: str = "/mnt/models/model.pkl"
    max_feature_vals: str = "src/models/articlequality/data/feature_values.tsv"
    force_http: bool = False
    model_name_v2: str = "articlequality_v2"
    model_path_v2: str = "/mnt/models/catboost_model.cbm"


settings = Settings()

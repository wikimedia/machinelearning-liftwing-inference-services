from pydantic_settings import BaseSettings


"""
Config class for edit-check project.
Reads variables as environment variables.
If environment variables do not exist, it uses the default values.
"""


class Settings(BaseSettings):
    max_batch_size: int = 100
    model_name: str = "edit-check"
    max_char_length: int = 1000


settings = Settings()

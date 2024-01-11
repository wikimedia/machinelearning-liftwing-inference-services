import os
import importlib


def select_models():
    model = os.environ.get("MODEL", None)
    if model:
        _models = importlib.import_module(f"models.{model}")
        for attr in dir(_models):
            if not attr.startswith("_"):  # Skip internal attributes
                globals()[attr] = getattr(_models, attr)
    else:
        import models  # noqa

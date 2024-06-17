import logging
import os
from typing import Any, Dict

import kserve
from kserve.errors import InferenceError

from python.preprocess_utils import validate_json_input
from utils import (
    get_article_html,
    get_article_features,
    normalize_features,
    load_quality_max_featurevalues,
)


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleQualityModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.pkl")
        self.load()

    def load(self) -> None:
        # TODO: Load the model from the model_path

        # Load the table of max feature values - use for feature normalization
        self.max_qual_vals = load_quality_max_featurevalues(
            "data/max-vals-html-dumps-ar-en-fr-hu-tr-zh.tsv"
        )
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        article_html = get_article_html(lang, rev_id)
        article_features = get_article_features(article_html)
        if article_features[0] > 0:  # page_length > 0
            normalized_features = normalize_features(
                self.max_qual_vals, lang, *article_features
            )
        else:
            raise InferenceError(
                f"Article with language {lang} and revid {rev_id}"
                " has errors when preprocessing"
            )

        inputs["normalized_features"] = normalized_features
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        # TODO: Run model.predict() on the preprocessed inputs
        return request


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = ArticleQualityModel(name=model_name)
    kserve.ModelServer(workers=1).start([model])

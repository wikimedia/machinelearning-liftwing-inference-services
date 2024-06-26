import logging
import os
from distutils.util import strtobool
from typing import Any, Dict
import pickle

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
    def __init__(
        self,
        name: str,
        model_path: str,
        max_feature_vals: str,
        force_http: bool = False,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = model_path
        self.max_feature_vals = max_feature_vals
        self.protocol = "http" if force_http else "https"
        self.load()

    def load(self) -> None:
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        # Load the table of max feature values - use for feature normalization
        self.max_qual_vals = load_quality_max_featurevalues(self.max_feature_vals)
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        article_html = get_article_html(lang, rev_id, self.protocol)
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
        predicted_value = self.model.predict([request["normalized_features"]])
        return predicted_value[0]


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.pkl")
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))
    max_feature_vals = os.environ.get(
        "MAX_FEATURE_VALS", "data/max-values-html-dumps-ar-en-fr-hu-tr-zh.tsv"
    )
    model = ArticleQualityModel(
        name=model_name,
        model_path=model_path,
        max_feature_vals=max_feature_vals,
        force_http=force_http,
    )
    kserve.ModelServer(workers=1).start([model])

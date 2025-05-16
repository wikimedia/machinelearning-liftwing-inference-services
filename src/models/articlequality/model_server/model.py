import logging
import math
from distutils.util import strtobool
from typing import Any, Dict, Tuple

import kserve
import numpy as np
from kserve import ModelServer
from kserve.errors import InferenceError
from statsmodels.iolib.smpickle import load_pickle
from utils import (
    get_article_features,
    get_article_html,
    load_quality_max_featurevalues,
    normalize_features,
)

from python.decorators import preprocess_size_bytes
from python.preprocess_utils import validate_json_input
from src.models.articlequality.model_server.config import Settings
from src.models.articlequality.model_server.model_v2 import ArticleQualityModelV2

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
        self.max_qual_vals = None
        self.model = None
        self.labels = ("Stub", "Start", "C", "B", "GA", "FA")
        self.feature_order = (
            "characters",
            "refs",
            "wikilinks",
            "categories",
            "media",
            "headings",
            "sources",
            "infobox",
            "messagebox",
        )
        self.load()
        self.top_score, self.score_range = self.extract_score_range()

    def load(self) -> None:
        self.model = load_pickle(self.model_path)
        # Load the table of max feature values - use for feature normalization
        self.max_qual_vals = load_quality_max_featurevalues(self.max_feature_vals)
        self.ready = True

    def extract_score_range(self) -> Tuple[float, float]:
        """Extract the top score and score range from the model."""
        top_input = []
        low_input = []
        for i, param_value in zip(self.feature_order, self.model.params):
            if param_value >= 0:
                top_input.append(1)
                low_input.append(0)
            else:  # negative coefficient
                top_input.append(0)
                low_input.append(1)
        top_score = self.model.predict(top_input, which="linpred")[0]
        low_score = self.model.predict(low_input, which="linpred")[0]
        top_score = top_score + 1  # maps maximum model output to 1
        score_range = (
            top_score - low_score
        )  # helps transform rest of outputs between 0 and 1
        return top_score, score_range

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        article_html = await get_article_html(lang, rev_id, self.protocol)
        raw_features = get_article_features(article_html)
        raw_features_dict = dict(zip(self.feature_order, raw_features))
        page_length_idx = self.feature_order.index("characters")
        if raw_features[page_length_idx] > 0:  # page_length > 0
            normalized_features_tuple = normalize_features(
                self.max_qual_vals, lang, *raw_features
            )
            normalized_features_dict = dict(
                zip(self.feature_order, normalized_features_tuple)
            )
        else:
            raise InferenceError(
                f"Article with language {lang} and revid {rev_id}"
                " has errors when preprocessing"
            )
        inputs["features"] = {
            "raw": raw_features_dict,
            "normalized": normalized_features_dict,
        }
        inputs["normalized_features"] = normalized_features_tuple
        return inputs

    @preprocess_size_bytes("articlequality", key_name="features")
    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        output = {}
        features = request["features"]
        extended_output = strtobool(request.get("extended_output", "False"))
        if extended_output:
            probs = self.model.predict(request["normalized_features"])
            label = self.labels[np.argmax(probs)]
            output["label"] = label
        raw_score = self.model.predict(request["normalized_features"], which="linpred")[
            0
        ]
        # normalize the score to be approximately between 0 and 1
        normalized_score = 1 - math.log(self.top_score - raw_score, self.score_range)
        if extended_output:
            output["features"] = features
        output["score"] = normalized_score
        output.update(
            {
                "model_name": "articlequality",
                "model_version": "1",  # model version should come directly from the model
                "wiki_db": f"{request.get('lang')}wiki",
                "revision_id": request.get("rev_id"),
            }
        )

        return output


if __name__ == "__main__":
    settings = Settings()
    model = ArticleQualityModel(
        name=settings.model_name,
        model_path=settings.model_path,
        max_feature_vals=settings.max_feature_vals,
        force_http=settings.force_http,
    )
    model_v2 = ArticleQualityModelV2(
        name=settings.model_name_v2,
        model_path=settings.model_path_v2,
        force_http=settings.force_http,
    )
    server = ModelServer()
    server.start([model, model_v2])

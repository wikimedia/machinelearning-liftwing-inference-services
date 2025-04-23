import logging
from typing import Any, Dict, List

from catboost import CatBoostClassifier

import kserve
from kserve.errors import InferenceError
from python.preprocess_utils import validate_json_input
from python.decorators import preprocess_size_bytes
from src.models.articlequality.model_server.request_model import RequestModel
from src.models.articlequality.model_server.utils import (
    get_article_features_v2,
    get_article_html,
)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleQualityModelV2(kserve.Model):
    def __init__(
        self,
        name: str,
        model_path: str,
        force_http: bool = False,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = model_path
        self.protocol = "http" if force_http else "https"
        self.model = None
        self.labels = None
        self.ordinal_class_weights = None
        self.feature_order = (
            "characters",
            "refs",
            "wikilinks",
            "categories",
            "media",
            "headings",
            "images",
            "first_paragraph_length",
        )
        self.load()

    def load(self) -> None:
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)
        self.labels = self.model.get_param("class_names")
        num_labels = len(self.labels)
        self.ordinal_class_weights = [i / (num_labels - 1) for i in range(num_labels)]
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        validated_inputs: Dict[str, List[Dict[str, Any]]] = validate_json_input(inputs)
        request_model = RequestModel(**validated_inputs)
        features_dict: Dict[str, List[Dict[str, Any]]] = {"features": []}
        for validated_input in request_model.instances:
            lang = validated_input.lang
            rev_id = validated_input.rev_id
            article_html = await get_article_html(lang, rev_id, self.protocol)
            features = get_article_features_v2(article_html)
            page_length_idx = self.feature_order.index("characters")
            if features[page_length_idx] <= 0:  # page_length > 0
                raise InferenceError(
                    f"Article with language {lang} and revid {rev_id}"
                    " has errors when preprocessing"
                )

            features_dict["features"].append(features)
        return features_dict

    @preprocess_size_bytes("articlequality", key_name="features")
    def predict(
        self, request: Dict[str, List[Dict[str, Any]]], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Returns following attributes for each input in predictions list:
        index: Index of the input.
        label: classification label.
        score: single quality score.
                Score is the weighted sum of the prediction probabilities between 0 and 1.
                We have checked distributions of the prediction probabilities for each predicted label.
                Experimentally, we see the model learns the order of the labels, e.g.
                    - Start, C probabilities are higher than the rest except for Stub when Stub is the label.
                    - GA, B probabilities are higher than the rest except for FA when the label is FA.
        prediction_probabilities: prediction probabilities for each class.
        features: calculated features for the given article.

        """
        features = request["features"]
        response = {"predictions": []}
        predictions = self.model.predict_proba(features)
        for feature, prediction_proba in zip(features, predictions):
            response["predictions"].append(
                {
                    "label": self.labels[prediction_proba.argmax()],
                    "score": sum(
                        p * w
                        for p, w in zip(prediction_proba, self.ordinal_class_weights)
                    ),
                    "prediction_probabilities": dict(
                        zip(self.labels, prediction_proba.tolist())
                    ),
                    "features": dict(zip(self.feature_order, feature)),
                }
            )
        response.update(
            {
                "model_name": "articlequality",
                "model_version": "2",  # model version should come directly from the model
            }
        )
        return response

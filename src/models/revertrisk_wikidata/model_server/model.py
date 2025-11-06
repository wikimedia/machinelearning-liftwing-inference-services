import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from typing import Any, Optional

import catboost as catb
import joblib
import kserve
import mwapi
import numpy as np
import pandas as pd
import transformers
from aiohttp import ClientSession, ClientTimeout
from kserve.errors import InferenceError, InvalidInput
from utils import (
    fetch_labels_from_api,
    parse_wikidata_revision_difference,
    prepare_input_for_bert,
    process_alteration,
    process_change,
    process_transformer_predictions,
)

from python.preprocess_utils import check_input_param, validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


# This class is needed for joblib to unpickle the model
@dataclass
class RevertRiskWikidataGraph2TextModelForLoad:
    metadata_classifier: catb.CatBoostClassifier
    text_classifier: transformers.Pipeline
    id2label: dict[str, str]
    model_version: int


class RevertRiskWikidataModel(kserve.Model):
    def __init__(
        self, name: str, model_path: str, force_http: bool, aiohttp_client_timeout: int
    ):
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.force_http = force_http
        self.aiohttp_client_timeout = aiohttp_client_timeout
        self.http_client_session: Optional[ClientSession] = None
        self.custom_user_agent = (
            "WMF ML Team revertrisk-wikidata model inference (LiftWing)"
        )
        self.ready = False
        self.id2label = {}
        self.load()

    def create_mwapi_session(self):
        """
        Create a new mwapi.AsyncSession with the correct protocol and user agent.
        """
        protocol = "http" if self.force_http else "https"
        host = f"{protocol}://www.wikidata.org"
        return mwapi.AsyncSession(host, user_agent=self.custom_user_agent)

    def load(self):
        """
        Load the Graph2Text model and the id2label dictionary.
        """
        setattr(
            sys.modules["__main__"],
            "RevertRiskWikidataGraph2TextModel",
            RevertRiskWikidataGraph2TextModelForLoad,
        )
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            error_message = f"Failed to load model from {self.model_path}. Reason: {e}"
            logging.critical(error_message)
            raise InferenceError(error_message)
        self.ready = True

    def get_http_client_session(self) -> ClientSession:
        """
        Get the aiohttp client session, create it if it doesn't exist.
        Unlike other model-server's, we want to reuse the same session across requests
        because this model-server uses one host (Wikidata) for all requests.
        """
        if self.http_client_session is None or self.http_client_session.closed:
            timeout = ClientTimeout(total=self.aiohttp_client_timeout)
            self.http_client_session = ClientSession(timeout=timeout)
        return self.http_client_session

    def _get_bert_scores(self, texts: list[str]) -> dict[str, float]:
        """
        Calculate BERT scores for the diffs.
        """
        if not texts:
            return {"mean": -999.0, "max": -999.0}

        tokenizer_kwargs = {"truncation": True, "max_length": 512}
        texts_to_process = prepare_input_for_bert(texts)
        predictions = self.model.text_classifier(
            texts_to_process, top_k=None, **tokenizer_kwargs, batch_size=8
        )
        scores = process_transformer_predictions(predictions)

        return {
            "mean": np.mean(scores) if scores else -999.0,
            "max": np.max(scores) if scores else -999.0,
        }

    async def _get_revision_content(
        self, session: mwapi.AsyncSession, rev_id: int
    ) -> tuple[str, Optional[int], str]:
        """
        Fetch the content, parent ID, and page title of a revision.
        """
        try:
            rev_doc = await session.get(
                action="query",
                prop="revisions",
                revids=rev_id,
                rvprop="content|ids|title",
                rvslots="main",
            )
        except Exception as e:
            error_message = f"Failed to fetch revision content in _get_revision_content for rev_id {rev_id}. Reason: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)
        page = list(rev_doc["query"]["pages"].values())[0]
        page_title = page["title"]
        current_rev = page["revisions"][0]
        if "slots" in current_rev:
            content = current_rev["slots"]["main"]["*"]
        else:
            content = current_rev.get("*", "")
        parent_id = current_rev.get("parentid")
        return content, parent_id, page_title

    async def _fetch_metadata_features(
        self, rev_id: int, session: mwapi.AsyncSession
    ) -> dict[str, Any]:
        """
        Fetch all features for a given revision ID from the MediaWiki API.
        """
        features = {}
        try:
            rev_doc = await session.get(
                action="query",
                prop="revisions",
                revids=rev_id,
                rvprop="user|userid|timestamp|ids",
                format="json",
            )
        except Exception as e:
            error_message = f"Failed to fetch metadata features in _fetch_metadata_features for rev_id {rev_id}. Reason: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)
        page_id = list(rev_doc["query"]["pages"].keys())[0]
        revision = rev_doc["query"]["pages"][page_id]["revisions"][0]
        user_name = revision["user"]
        rev_timestamp = datetime.fromisoformat(
            revision["timestamp"].replace("Z", "+00:00")
        )
        parent_id = revision.get("parentid", 0)

        user_doc = await session.get(
            action="query",
            list="users",
            ususers=user_name,
            usprop="groups|registration",
            format="json",
        )
        user = user_doc["query"]["users"][0]
        features["user_is_anonymous"] = str("userid" not in user)
        features["user_is_bot"] = str(int("bot" in user.get("groups", [])))
        user_groups = user.get("groups", [])

        for group in self.model.metadata_classifier.feature_names_:
            if group.startswith("event_user_groups-"):
                features[group] = str(float(group.split("-")[1] in user_groups))

        if user.get("registration"):
            reg_timestamp = datetime.fromisoformat(
                user["registration"].replace("Z", "+00:00")
            )
            features["user_age"] = (rev_timestamp - reg_timestamp).total_seconds() / (
                60 * 60 * 24
            )
        else:
            features["user_age"] = 0

        if parent_id != 0:
            parent_rev_doc = await session.get(
                action="query",
                prop="revisions",
                revids=parent_id,
                rvprop="timestamp",
                format="json",
            )
            parent_revision = list(parent_rev_doc["query"]["pages"].values())[0][
                "revisions"
            ][0]
            parent_rev_timestamp = datetime.fromisoformat(
                parent_revision["timestamp"].replace("Z", "+00:00")
            )
            features["page_seconds_since_previous_revision"] = (
                rev_timestamp - parent_rev_timestamp
            ).total_seconds()
        else:
            features["page_seconds_since_previous_revision"] = 0

        first_rev_doc = await session.get(
            action="query",
            prop="revisions",
            pageids=page_id,
            rvdir="newer",
            rvlimit=1,
            rvprop="timestamp",
            format="json",
        )
        first_revision = first_rev_doc["query"]["pages"][page_id]["revisions"][0]
        first_rev_timestamp = datetime.fromisoformat(
            first_revision["timestamp"].replace("Z", "+00:00")
        )
        features["page_age"] = (rev_timestamp - first_rev_timestamp).total_seconds() / (
            60 * 60 * 24
        )

        return features

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        """
        Preprocess the input request to fetch features.
        """
        inputs = validate_json_input(inputs)
        rev_id = inputs.get("rev_id")
        if not isinstance(rev_id, int) or rev_id <= 0:
            error_message = (
                f"Invalid rev_id: {rev_id}. The rev_id must be a positive integer."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)
        check_input_param(rev_id=rev_id)

        session = self.create_mwapi_session()
        try:
            current_text, parent_id, page_title = await self._get_revision_content(
                session, rev_id
            )
            parent_text = ""
            if parent_id:
                parent_text, _, _ = await self._get_revision_content(session, parent_id)

            diffs = parse_wikidata_revision_difference(parent_text, current_text)

            metadata_features = await self._fetch_metadata_features(rev_id, session)
        except Exception as e:
            error_message = f"Error in preprocess for rev_id {rev_id}. Reason: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)
        finally:
            await session.session.close()

        return {
            "rev_id": rev_id,
            "page_title": page_title,
            "diffs": diffs,
            "metadata_features": metadata_features,
        }

    async def predict(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        """
        Make a prediction with the model.
        """
        rev_id = inputs["rev_id"]
        page_title = inputs["page_title"]
        diffs = inputs["diffs"]
        metadata_features = inputs["metadata_features"]

        # Extract all Q/P IDs from diffs for label fetching
        def extract_entity_ids(diffs):
            ids = set()
            for diff_str in diffs:
                matches = re.findall(r"Q\d+|P\d+", diff_str)
                ids.update(matches)
            return list(ids)

        entity_ids = extract_entity_ids(diffs)
        session = self.create_mwapi_session()
        try:
            labels_dict = await fetch_labels_from_api(session, entity_ids)
        finally:
            await session.session.close()

        add_text = process_alteration(
            left_q_id=page_title,
            alterations=diffs[0],
            action_type="add: ",
            labels_dict=labels_dict,
        )
        remove_text = process_alteration(
            left_q_id=page_title,
            alterations=diffs[1],
            action_type="remove: ",
            labels_dict=labels_dict,
        )
        change_text = process_change(
            left_q_id=page_title,
            changes=diffs[2],
            action_type="change: ",
            labels_dict=labels_dict,
        )

        add_scores = self._get_bert_scores(add_text)
        remove_scores = self._get_bert_scores(remove_text)
        change_scores = self._get_bert_scores(change_text)

        features = {
            "add_score_mean": add_scores["mean"],
            "add_score_max": add_scores["max"],
            "remove_score_mean": remove_scores["mean"],
            "remove_score_max": remove_scores["max"],
            "change_score_mean": change_scores["mean"],
            "change_score_max": change_scores["max"],
            **metadata_features,
        }

        X = []
        cat_feature_indices = self.model.metadata_classifier.get_cat_feature_indices()
        for i, feature_name in enumerate(self.model.metadata_classifier.feature_names_):
            if pd.isna(features.get(feature_name)):
                X.append("NaN" if i in cat_feature_indices else -999)
            else:
                X.append(features[feature_name])

        [_, prob_yes] = self.model.metadata_classifier.predict_proba(X)

        return {
            "model_name": self.name,
            "model_version": str(self.model.model_version),
            "revision_id": rev_id,
            "output": {
                "prediction": bool(prob_yes > 0.5),
                "probabilities": {
                    "true": float(prob_yes),
                    "false": float(1 - prob_yes),
                },
            },
        }


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "revertrisk-wikidata")
    model_path = os.environ.get(
        "MODEL_PATH", "/mnt/models/wikidata_revertrisk_graph2text_v2.pkl"
    )
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))
    aiohttp_client_timeout = int(os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5))
    model = RevertRiskWikidataModel(
        name=model_name,
        model_path=model_path,
        force_http=force_http,
        aiohttp_client_timeout=aiohttp_client_timeout,
    )
    kserve.ModelServer(workers=1).start([model])

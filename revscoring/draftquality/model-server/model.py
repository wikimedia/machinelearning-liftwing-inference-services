import bz2
import os
from http import HTTPStatus
from typing import Dict

import aiohttp
import kserve
import mwapi
import requests
import tornado.web
from revscoring import Model
from revscoring.errors import RevisionNotFound
from revscoring.extractors import api
from revscoring.features import trim

import events
import preprocess_utils
import extractor_utils


class DraftqualityModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.FEATURE_VAL_KEY = "feature_values"
        self.EXTENDED_OUTPUT_KEY = "extended_output"
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        self.CUSTOM_UA = "WMF ML team draftquality model"
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"

    def load(self):
        with bz2.open("/mnt/models/model.bz2") as f:
            self.model = Model.load(f)
        self.ready = True

    async def preprocess(self, inputs: Dict) -> Dict:
        """Retrieve features from mediawiki."""
        rev_id = preprocess_utils.get_rev_id(inputs, self.REVISION_CREATE_EVENT_KEY)
        # The predict() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = preprocess_utils.get_revision_event(
            inputs, self.REVISION_CREATE_EVENT_KEY
        )
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")

        async with aiohttp.ClientSession() as s:
            mw_http_cache = await extractor_utils.get_revscoring_extractor_cache(
                rev_id,
                self.CUSTOM_UA,
                s,
                wiki_url=wiki_url,
                wiki_host=wiki_host,
                fetch_extra_info=True,
            )

            self.extractor = api.Extractor(
                mwapi.Session(wiki_url, user_agent=self.CUSTOM_UA),
                http_cache=mw_http_cache,
            )
            inputs[self.FEATURE_VAL_KEY] = self._fetch_draftquality_features(rev_id)
            if extended_output:
                base_feature_values = self.extractor.extract(
                    rev_id, list(trim(self.model.features))
                )
                inputs[self.EXTENDED_OUTPUT_KEY] = {
                    str(f): v
                    for f, v in zip(
                        list(trim(self.model.features)), base_feature_values
                    )
                }
            return inputs

    async def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        self.prediction_results = self.model.score(feature_values)
        if extended_output:
            # add extended output to reach feature parity with ORES, like:
            # https://ores.wikimedia.org/v3/scores/enwiki/1083325118/draftquality?features
            # If only rev_id is given in input.json, only the prediction results
            # will be present in the response. If the extended_output flag is true,
            # features output will be included in the response.
            output = {
                "predictions": self.prediction_results,
                "features": extended_output,
            }
        else:
            output = {
                "predictions": self.prediction_results,
            }
        # Send a revision-score event to EventGate, generated from
        # the revision-create event passed as input.
        if self.revision_create_event:
            revision_score_event = events.generate_revision_score_event(
                self.revision_create_event,
                self.EVENTGATE_STREAM,
                self.model.version,
                self.prediction_results,
                "draftquality",
            )
            await events.send_event(
                revision_score_event,
                self.EVENTGATE_URL,
                self.TLS_CERT_BUNDLE_PATH,
                self.CUSTOM_UA,
                self._http_client,
            )
        return output

    def _fetch_draftquality_features(self, rev_id: int) -> Dict:
        """Retrieve draftquality features."""
        try:
            feature_values = list(self.extractor.extract(rev_id, self.model.features))
        except RevisionNotFound:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Revision {} not found".format(rev_id),
            )
        return feature_values


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = DraftqualityModel(inference_name)
    model.load()
    kserve.ModelServer(workers=1).start([model])

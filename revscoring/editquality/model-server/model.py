import json
import logging
import os
from http import HTTPStatus
from typing import Dict, Optional

import kserve
import mwapi
import requests
import tornado.web
import tornado.httpclient
from revscoring import Model
from revscoring.errors import RevisionNotFound
from revscoring.extractors import api
from revscoring.features import trim

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class EditQualityModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.FEATURE_VAL_KEY = "feature_values"
        self.EXTENDED_OUTPUT_KEY = "extended_output"
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        self.CUSTOM_UA = "WMF ML Team editquality model svc"
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = self._get_rev_id(inputs)
        # The postprocess() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = self._get_revision_event(inputs)
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")

        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None

        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent=self.CUSTOM_UA, session=s)
        )
        inputs[self.FEATURE_VAL_KEY] = self.fetch_editquality_features(rev_id)
        if extended_output:
            base_feature_values = self.extractor.extract(
                rev_id, list(trim(self.model.features))
            )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v
                for f, v in zip(list(trim(self.model.features)), base_feature_values)
            }
        return inputs

    def _get_revision_event(self, inputs: Dict) -> Optional[str]:
        try:
            return inputs[self.REVISION_CREATE_EVENT_KEY]
        except KeyError:
            return None

    def _get_rev_id(self, inputs: Dict) -> Dict:
        try:
            # If a revision create event is passed as input,
            # its rev-id is considerate the one to score.
            # Otherwise, we look for a specific "rev_id" input.
            if self.REVISION_CREATE_EVENT_KEY in inputs.keys():
                rev_id = inputs[self.REVISION_CREATE_EVENT_KEY]["rev_id"]
            else:
                rev_id = inputs["rev_id"]
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "rev_id" in input data.',
            )
        if not isinstance(rev_id, int):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "rev_id" to be an integer.',
            )
        return rev_id

    def fetch_editquality_features(self, rev_id: int) -> Dict:
        """Retrieve editquality features."""
        try:
            feature_values = list(self.extractor.extract(rev_id, self.model.features))
        except RevisionNotFound:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Revision {} not found".format(rev_id),
            )

        return feature_values

    def get_revision_score_event(self, rev_create_event) -> Dict:
        if "goodfaith" in self.name:
            model_name = "goodfaith"
        elif "damaging" in self.name:
            model_name = "damaging"
        else:
            model_name = "reverted"
        return {
            "$schema": "/mediawiki/revision/score/2.0.0",
            "meta": {
                "stream": self.EVENTGATE_STREAM,
            },
            "database": rev_create_event["database"],
            "page_id": rev_create_event["page_id"],
            "page_title": rev_create_event["page_title"],
            "page_namespace": rev_create_event["page_namespace"],
            "page_is_redirect": rev_create_event["page_is_redirect"],
            "performer": rev_create_event["performer"],
            "rev_id": rev_create_event["rev_id"],
            "rev_parent_id": rev_create_event["rev_parent_id"],
            "rev_timestamp": rev_create_event["rev_timestamp"],
            model_name: {
                "model_name": model_name,
                "model_version": self.model.version,
                "predictions": self.prediction_results,
            },
        }

    async def send_event(self, revision_create_event):
        # TODO: check if the revision_create_event is well formed,
        # maybe checking the schema and report an error if not revision/create.
        revision_score_event = self.get_revision_score_event(revision_create_event)
        try:
            response = await self._http_client.fetch(
                self.EVENTGATE_URL,
                method="POST",
                ca_certs=self.TLS_CERT_BUNDLE_PATH,
                body=json.dumps(revision_score_event),
                headers={"Content-type": "application/json"},
                user_agent=os.environ.get("CUSTOM_UA"),
            )
            logging.debug(
                "Successfully sent the following event to "
                "EventGate: {}".format(revision_score_event)
            )
        except tornado.httpclient.HTTPError as e:
            logging.error(
                "The revision score event has been rejected by EventGate, "
                "that returned a non-200 HTTP return code "
                "with the following error: {}".format(e)
            )
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason=(
                    "Eventgate rejected the revision-score event "
                    "(a non-HTTP-200 response was returned). "
                    "Please contact the ML team for more info."
                ),
            )
        except Exception as e:
            logging.error(
                "Unexpected error while trying to send a revision score "
                "event to EventGate: {}".format(e)
            )
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason=(
                    "An unexpected error has occurred while trying "
                    "to send the revision-score event to Eventgate. "
                    "Please contact the ML team for more info."
                ),
            )

    async def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        self.prediction_results = self.model.score(feature_values)
        if extended_output:
            # add extended output to reach feature parity with ORES, like:
            # https://ores.wikimedia.org/v3/scores/enwiki/186357639/goodfaith?features
            # If only rev_id is given in input.json, only the prediction results
            # will be present in the response. If the extended_output flag is true,
            # features output will be included in the response.
            output = {
                "predictions": self.prediction_results,
                "features": extended_output,
            }
        else:
            output = {"predictions": self.prediction_results}
        # Send a revision-score event to EventGate, generated from
        # the revision-create event passed as input.
        if self.revision_create_event:
            await self.send_event(self.revision_create_event)
        return output


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = EditQualityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

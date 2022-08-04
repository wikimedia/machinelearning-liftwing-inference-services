import articlequality
import kserve
import os
from revscoring import Model
from typing import Dict
import requests
import mwapi
import tornado.web
from http import HTTPStatus
from revscoring.extractors import api
from revscoring.features import trim

import events
import preprocess_utils


class ArticlequalityModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.FEATURE_VAL_KEY = "article_text"
        self.EXTENDED_OUTPUT_KEY = "extended_output"
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        self.CUSTOM_UA = "WMF ML team articlequality transformer"
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    def preprocess(self, inputs: Dict) -> Dict:
        """Fetch article text"""
        rev_id = preprocess_utils.get_rev_id(inputs, self.REVISION_CREATE_EVENT_KEY)
        # The predict() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = preprocess_utils.get_revision_event(
            inputs, self.REVISION_CREATE_EVENT_KEY
        )
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")
        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None
        self.session = mwapi.Session(wiki_url, user_agent=self.CUSTOM_UA, session=s)
        inputs[self.FEATURE_VAL_KEY] = self._fetch_articlequality_text(rev_id)
        self.extractor = api.Extractor(self.session)
        if extended_output:
            base_feature_values = self.extractor.extract(
                rev_id, list(trim(self.model.features))
            )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v
                for f, v in zip(list(trim(self.model.features)), base_feature_values)
            }
        return inputs

    async def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        self.prediction_results = articlequality.score(self.model, feature_values)
        if extended_output:
            # add extended output to reach feature parity with ORES, like:
            # https://ores.wikimedia.org/v3/scores/enwiki/186357639/articlequality?features
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
                "articlequality",
            )
            await events.send_event(
                revision_score_event,
                self.EVENTGATE_URL,
                self.TLS_CERT_BUNDLE_PATH,
                self.CUSTOM_UA,
                self._http_client,
            )
        return output

    def _fetch_articlequality_text(self, rev_id: int) -> Dict:
        """Retrieve article text features."""
        doc = self.session.get(
            action="query",
            prop="revisions",
            revids=[rev_id],
            rvprop=["ids", "content"],
            rvslots=["main"],
            formatversion=2,
        )
        try:
            rev_doc = doc["query"]["pages"][0]["revisions"][0]
        except (KeyError, IndexError) as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Revision {} not found.".format(rev_id),
            )
        content = rev_doc["slots"]["main"].get("content")
        return content


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = ArticlequalityModel(inference_name)
    model.load()
    kserve.ModelServer(workers=1).start([model])

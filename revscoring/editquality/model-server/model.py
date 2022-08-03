import asyncio
import json
import logging
import os
from http import HTTPStatus
from typing import Dict, Optional

import aiohttp
import kserve
import mwapi
import requests
import tornado.web
import tornado.httpclient
from revscoring import Model
from revscoring.errors import RevisionNotFound
from revscoring.extractors import api
from revscoring.extractors.api import MWAPICache
from revscoring.features import trim
from mwapi.errors import (
    APIError,
    ConnectionError,
    RequestError,
    TimeoutError,
    TooManyRedirectsError,
)

import events

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class EditQualityModel(kserve.Model):
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

    async def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = self._get_rev_id(inputs)
        # The postprocess() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = self._get_revision_event(inputs)
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")

        async with aiohttp.ClientSession() as s:
            if wiki_host:
                s.headers.update({"Host": wiki_host})

            session = mwapi.AsyncSession(wiki_url, user_agent=self.CUSTOM_UA, session=s)

            # Get all info from MediaWiki API.
            # The revscoring API extractor can automatically fetch
            # data from the MW API as well, but sadly only with blocking
            # IO (namely, using Session from the mwapi package).
            # Since KServe works with Tornado and asyncio, we prefer
            # to use mwapi's AsyncSession and pass the data (as MWAPICache)
            # to revscoring.
            # From tests in T309623, the API extractor fetches:
            # - info related to the rev-id
            # - user and parent-rev-id data as well
            # The total is 3 MW API calls.
            params = {
                "rvprop": {
                    "content",
                    "userid",
                    "size",
                    "contentmodel",
                    "ids",
                    "user",
                    "comment",
                    "timestamp",
                }
            }
            try:
                rev_id_doc = await asyncio.create_task(
                    session.get(
                        action="query",
                        prop="revisions",
                        revids=[rev_id],
                        rvslots="main",
                        **params
                    )
                )

                # The output returned by the MW API is a little
                # convoluted and probably meant for batches of rev-ids.
                # In our case we fetch only one rev-id at the time,
                # so we can use assumptions about how many elements
                # there will be in the results.
                try:
                    revision_info = list(rev_id_doc.get("query").get("pages").values())[
                        0
                    ]["revisions"][0]
                except Exception as e:
                    logger.error(
                        "The rev-id doc retrieved from the MW API "
                        "does not contain all the data needed "
                        "to extract features properly. "
                        "The error is {} and the document is: {}".format(e, rev_id_doc)
                    )
                    raise tornado.web.HTTPError(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        reason=(
                            "The rev-id doc retrieved from the MW API "
                            "does not contain all the data needed "
                            "to extract features properly. Please contact the ML-Team if the issue persists."
                        ),
                    )

                parent_rev_id = revision_info.get("parentid")
                user = revision_info.get("user")
                user_params = {
                    "usprop": {"groups", "registration", "editcount", "gender"}
                }

                parent_rev_id_doc, user_doc = await asyncio.gather(
                    session.get(
                        action="query",
                        prop="revisions",
                        revids=[parent_rev_id],
                        rvslots="main",
                        **params
                    ),
                    session.get(
                        action="query", list="users", ususers=[user], **user_params
                    ),
                )
            except (
                APIError,
                ConnectionError,
                RequestError,
                TimeoutError,
                TooManyRedirectsError,
            ) as e:
                logging.error(
                    "An error has occurred while fetching feature "
                    "values from the MW API: {}".format(e)
                )
                raise tornado.web.HTTPError(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    reason=(
                        "An error happened while fetching feature values from "
                        "the MediaWiki API, please contact the ML-Team "
                        "if the issue persists."
                    ),
                )

            # Populate the MWAPICache
            http_cache = MWAPICache()
            http_cache.add_revisions_batch_doc([rev_id], rev_id_doc)
            http_cache.add_revisions_batch_doc([parent_rev_id], parent_rev_id_doc)
            http_cache.add_users_batch_doc([user], user_doc)

            # Call the extractor with the MWAPICache
            self.extractor = api.Extractor(
                mwapi.Session(wiki_url, user_agent=self.CUSTOM_UA, session=s),
                http_cache=http_cache,
            )
            inputs[self.FEATURE_VAL_KEY] = self.fetch_editquality_features(rev_id)
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
        return events.generate_revision_score_event(
            rev_create_event,
            self.EVENTGATE_STREAM,
            self.model.version,
            self.prediction_results,
            model_name,
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
            revision_score_event = self.get_revision_score_event(
                self.revision_create_event
            )
            await events.send_event(
                revision_score_event,
                self.EVENTGATE_URL,
                self.TLS_CERT_BUNDLE_PATH,
                self.CUSTOM_UA,
                self._http_client,
            )
        return output


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = EditQualityModel(inference_name)
    model.load()
    kserve.ModelServer(workers=1).start([model])

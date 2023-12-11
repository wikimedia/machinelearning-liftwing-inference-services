import bz2
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp
import kserve
import mwapi
from kserve.errors import InvalidInput
from python.preprocess_utils import validate_json_input
from revscoring.extractors import api
from revscoring.features import trim

from python import events, logging_utils
from revscoring import Model
from revscoring_model.model_servers import extractor_utils

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class RevscoringModelType(Enum):
    ARTICLEQUALITY = "articlequality"
    ARTICLETOPIC = "articletopic"
    DRAFTQUALITY = "draftquality"
    DRAFTTOPIC = "drafttopic"
    EDITQUALITY_DAMAGING = "damaging"
    EDITQUALITY_GOODFAITH = "goodfaith"
    EDITQUALITY_REVERTED = "reverted"
    ITEMQUALITY = "itemquality"
    ITEMTOPIC = "itemtopic"

    @classmethod
    def get_model_type(cls, inference_name: str):
        """
        Lookup function that searches for the model type value in the inference service name.
        e.g. searches for 'articlequality` in `enwiki-articlequality`
        """
        for _, model in cls.__members__.items():
            if model.value in inference_name:
                return model
        raise LookupError(
            f"INFERENCE_NAME '{inference_name}' could not be matched to a revscoring model type."
        )


class RevscoringModel(kserve.Model):
    def __init__(self, name: str, model_kind: RevscoringModelType):
        super().__init__(name)
        self.name = name
        self.model_kind = model_kind
        self.ready = False
        self.wiki_url = self._get_wiki_url()
        self.FEATURE_VAL_KEY = "feature_values"
        self.EXTENDED_OUTPUT_KEY = "extended_output"
        self.EVENT_KEY = "event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self.CUSTOM_UA = f"WMF ML Team {model_kind.value} model svc"
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self._http_client_session = {}
        if model_kind in [
            RevscoringModelType.EDITQUALITY_DAMAGING,
            RevscoringModelType.EDITQUALITY_GOODFAITH,
            RevscoringModelType.EDITQUALITY_REVERTED,
            RevscoringModelType.DRAFTQUALITY,
        ]:
            self.extra_mw_api_calls = True
        else:
            self.extra_mw_api_calls = False
        self.model_path = self.get_model_path()
        self.model = self.load()
        self.ready = True
        self.prediction_results = None
        # FIXME: this may not be needed, in theory we could simply rely on
        # kserve.constants.KSERVE_LOGLEVEL (passing KSERVE_LOGLEVEL as env var)
        # but it doesn't seem to work.
        logging_utils.set_log_level()

    def _get_wiki_url(self):
        if "WIKI_URL" not in os.environ:
            raise ValueError(
                "The environment variable WIKI_URL is not set. Please set it before running the server."
            )
        wiki_url = os.environ.get("WIKI_URL")
        return wiki_url

    def score(self, feature_values):
        return self.model.score(feature_values)

    def fetch_features(self, rev_id, features, extractor, cache):
        return extractor_utils.fetch_features(rev_id, features, extractor, cache)

    def get_http_client_session(self, endpoint):
        """Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one)."""
        timeout = aiohttp.ClientTimeout(total=self.AIOHTTP_CLIENT_TIMEOUT)
        if (
            self._http_client_session.get(endpoint, None) is None
            or self._http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self._http_client_session[endpoint] = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._http_client_session[endpoint]

    def get_model_path(self):
        if "MODEL_PATH" in os.environ:
            model_path = os.environ["MODEL_PATH"]
        elif self.model_kind == RevscoringModelType.DRAFTQUALITY:
            model_path = "/mnt/models/model.bz2"
        else:
            model_path = "/mnt/models/model.bin"
        return model_path

    def load(self):
        if self.model_kind == RevscoringModelType.DRAFTQUALITY:
            with bz2.open(self.model_path) as f:
                return Model.load(f)
        else:
            with open(self.model_path) as f:
                return Model.load(f)

    async def get_extractor(self, inputs, rev_id):
        # The postprocess() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = self.get_revision_event(inputs, self.EVENT_KEY)
        if self.revision_create_event:
            inputs["rev_id"] = rev_id
        wiki_host = os.environ.get("WIKI_HOST")

        # This is a workaround to allow the revscoring's extractor to leverage
        # aiohttp/asyncio HTTP calls. We inject a MW API cache later on in
        # the extractor, that in turn will not make any (blocking, old style)
        # HTTP calls via libs like requests.
        mw_http_cache = await extractor_utils.get_revscoring_extractor_cache(
            rev_id,
            self.CUSTOM_UA,
            self.get_http_client_session("mwapi"),
            wiki_url=self.wiki_url,
            wiki_host=wiki_host,
            fetch_extra_info=self.extra_mw_api_calls,
        )

        # Create the revscoring's extractor with the MWAPICache built above.
        return api.Extractor(
            mwapi.Session(self.wiki_url, user_agent=self.CUSTOM_UA),
            http_cache=mw_http_cache,
        )

    async def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        inputs = validate_json_input(inputs)

        rev_id = self.get_rev_id(inputs, self.EVENT_KEY)
        extended_output = inputs.get("extended_output", False)
        extractor = await self.get_extractor(inputs, rev_id)

        # The idea of this cache variable is to avoid extra cpu-bound
        # computations when executing fetch_features in the extended_output
        # branch. Revscoring allows to pass a cache parameter to save
        # info about { rev-id -> features } for subsequent calls.
        # We pass 'cache' for reference, so that fetch_features can populate/use
        # it if needed. This sadly doesn't work with a process pool, since
        # behind the scenes the work is done in another Python process
        # (and input/output is pickled/unpickled). The reference doesn't work
        # of course, and any attempt to return it from fetch_features led to
        # pickling errors. For the moment, until we solve the pickling errors
        # in revscoring (not sure if we want to do it), enabling extended_output
        # and using a process pool will mean recomputing fetch_features.
        cache = {}

        inputs[self.FEATURE_VAL_KEY] = self.fetch_features(
            rev_id, self.model.features, extractor, cache
        )

        if extended_output:
            bare_model_features = list(trim(self.model.features))
            base_feature_values = self.fetch_features(
                rev_id, bare_model_features, extractor, cache
            )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v for f, v in zip(bare_model_features, base_feature_values)
            }
        return inputs

    def get_revision_score_event(self, rev_create_event: Dict[str, Any]) -> Dict:
        return events.generate_revision_score_event(
            rev_create_event,
            self.EVENTGATE_STREAM,
            self.model.version,
            self.prediction_results,
            self.model_kind.value,
        )

    def get_output(self, request: Dict, extended_output: bool):
        wiki_db, model_name = self.name.split("-")
        rev_id = request.get("rev_id")
        output = {
            wiki_db: {
                "models": {model_name: {"version": self.model.version}},
                "scores": {rev_id: {model_name: {"score": self.prediction_results}}},
            }
        }
        if extended_output:
            # add extended output to reach feature parity with ORES, like:
            # https://ores.wikimedia.org/v3/scores/enwiki/186357639/goodfaith?features
            # If only rev_id is given in input.json, only the prediction results
            # will be present in the response. If the extended_output flag is true,
            # features output will be included in the response.
            output[wiki_db]["scores"][rev_id][model_name]["features"] = extended_output
        return output

    async def send_event(self) -> None:
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
                self.get_http_client_session("eventgate"),
            )

    def get_revision_event(self, inputs: Dict, event_input_key) -> Optional[str]:
        try:
            return inputs[event_input_key]
        except KeyError:
            return None

    def get_rev_id(self, inputs: Dict, event_input_key) -> Dict:
        """Get a revision id from the inputs provided.
        The revision id can be contained into an event dict
        or passed directly as value.
        """
        try:
            # If a revision event is passed as input,
            # its rev-id is considerate the one to score.
            # Otherwise, we look for a specific "rev_id" input.
            if event_input_key in inputs:
                if inputs[event_input_key]["$schema"].startswith(
                    "/mediawiki/revision/create/1"
                ) or inputs[event_input_key]["$schema"].startswith(
                    "/mediawiki/revision/create/2"
                ):
                    rev_id = inputs[event_input_key]["rev_id"]
                elif inputs[event_input_key]["$schema"].startswith(
                    "/mediawiki/page/change/1"
                ):
                    rev_id = inputs[event_input_key]["revision"]["rev_id"]
                else:
                    raise InvalidInput(
                        f"Unsupported event of schema {inputs[event_input_key]['$schema']}, "
                        "the rev_id value cannot be determined."
                    )
            else:
                rev_id = inputs["rev_id"]
        except KeyError:
            logging.error("Missing rev_id in input data.")
            raise InvalidInput('Expected "rev_id" in input data.')
        if not isinstance(rev_id, int):
            logging.error("Expected rev_id to be an integer.")
            raise InvalidInput('Expected "rev_id" to be an integer.')
        return rev_id

    async def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        self.prediction_results = self.score(feature_values)
        output = self.get_output(request, extended_output)
        await self.send_event()
        return output

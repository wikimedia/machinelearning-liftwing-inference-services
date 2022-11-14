import asyncio
import atexit
import bz2
import logging
import os
from concurrent.futures.process import BrokenProcessPool
from http import HTTPStatus
from typing import Dict
from enum import Enum

import aiohttp
import kserve
import mwapi
import tornado

from revscoring import Model
from revscoring.extractors import api
from revscoring.features import trim

import events
import extractor_utils
import logging_utils
import preprocess_utils
import process_utils

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class RevscoringModelType(Enum):
    EDITQUALITY = "editquality"
    DRAFTQUALITY = "draftquality"
    ITEMQUALITY = "itemquality"
    ARTICLETOPIC = "articletopic"
    ITEMTOPIC = "itemtopic"
    DRAFTTOPIC = "drafttopic"
    ARTICLEQUALITY = "articlequality"


class RevscoringModel(kserve.Model):
    def __init__(self, name: str, model_kind: RevscoringModelType):
        super().__init__(name)
        self.name = name
        self.model_kind = model_kind
        self.ready = False
        self.FEATURE_VAL_KEY = "feature_values"
        self.EXTENDED_OUTPUT_KEY = "extended_output"
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        self.CUSTOM_UA = f"WMF ML Team {model_kind} model svc"
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self._http_client_session = None
        atexit.register(self._shutdown)
        self.run_in_process_pool = False
        if os.environ.get("ASYNCIO_USE_PROCESS_POOL", "False") == "True":
            self.run_in_process_pool = True
            self.asyncio_aux_workers = os.environ.get("ASYNCIO_AUX_WORKERS")
            self.process_pool = process_utils.create_process_pool(
                self.asyncio_aux_workers
            )
        if model_kind in [
            RevscoringModelType.EDITQUALITY,
            RevscoringModelType.DRAFTQUALITY,
        ]:
            self.extra_mw_api_calls = True
        else:
            self.extra_mw_api_calls = False
        self.model = self.load()
        self.ready = True
        # FIXME: this may not be needed, in theory we could simply rely on
        # kserve.constants.KSERVE_LOGLEVEL (passing KSERVE_LOGLEVEL as env var)
        # but it doesn't seem to work.
        logging_utils.set_log_level()

    @property
    def http_client_session(self):
        if self._http_client_session is None or self._http_client_session.closed:
            logging.info("Opening a new Asyncio session.")
            self._http_client_session = aiohttp.ClientSession()
        return self._http_client_session

    def _shutdown(self):
        if self._http_client_session and not self._http_client_session.closed:
            logging.info("Closing asyncio session")
            asyncio.run(self._http_client_session.close())

    def load(self):
        if self.model_kind == RevscoringModelType.DRAFTQUALITY:
            with bz2.open("/mnt/models/model.bz2") as f:
                return Model.load(f)
        else:
            with open("/mnt/models/model.bin") as f:
                return Model.load(f)

    async def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = preprocess_utils.get_rev_id(inputs, self.REVISION_CREATE_EVENT_KEY)
        # The postprocess() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = preprocess_utils.get_revision_event(
            inputs, self.REVISION_CREATE_EVENT_KEY
        )
        if self.revision_create_event:
            inputs["rev_id"] = rev_id
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")

        # This is a workaround to allow the revscoring's extractor to leverage
        # aiohttp/asyncio HTTP calls. We inject a MW API cache later on in
        # the extractor, that in turn will not make any (blocking, old style)
        # HTTP calls via libs like requests.
        mw_http_cache = await extractor_utils.get_revscoring_extractor_cache(
            rev_id,
            self.CUSTOM_UA,
            self.http_client_session,
            wiki_url=wiki_url,
            wiki_host=wiki_host,
            fetch_extra_info=self.extra_mw_api_calls,
        )

        # Create the revscoring's extractor with the MWAPICache built above.
        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent=self.CUSTOM_UA),
            http_cache=mw_http_cache,
        )

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

        # The fetch_features function can be heavily cpu-bound, it depends
        # on the complexity of the rev-id to process. Running cpu-bound
        # code inside the asyncio/tornado eventloop will block the thread
        # and cause processing delays, so we use a process pool instead
        # (still enabled/disabled as opt-in).
        # See: https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools
        if self.run_in_process_pool:
            inputs[self.FEATURE_VAL_KEY] = await self._run_in_process_pool(
                extractor_utils.fetch_features,
                rev_id,
                self.model.features,
                self.extractor,
                cache,
            )
        else:
            inputs[self.FEATURE_VAL_KEY] = extractor_utils.fetch_features(
                rev_id, self.model.features, self.extractor, cache
            )

        if extended_output:
            bare_model_features = list(trim(self.model.features))
            if self.run_in_process_pool:
                base_feature_values = await self._run_in_process_pool(
                    extractor_utils.fetch_features,
                    rev_id,
                    bare_model_features,
                    self.extractor,
                    cache,
                )
            else:
                base_feature_values = extractor_utils.fetch_features(
                    rev_id, bare_model_features, self.extractor, cache
                )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v for f, v in zip(bare_model_features, base_feature_values)
            }
        return inputs

    def get_revision_score_event(self, rev_create_event) -> Dict:
        if self.model_kind == RevscoringModelType.EDITQUALITY:
            if "goodfaith" in self.name:
                model_name = "goodfaith"
            elif "damaging" in self.name:
                model_name = "damaging"
            else:
                model_name = "reverted"
        else:
            model_name = self.model_kind.value
        return events.generate_revision_score_event(
            rev_create_event,
            self.EVENTGATE_STREAM,
            self.model.version,
            self.prediction_results,
            model_name,
        )

    async def _run_in_process_pool(self, *args):
        try:
            return await process_utils.run_in_process_pool(self.process_pool, *args)
        except BrokenProcessPool as e:
            logging.exception("Re-creation of a newer process pool before proceeding.")
            self.process_pool = process_utils.refresh_process_pool(
                self.process_pool, self.asyncio_aux_workers
            )
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason=(
                    "An error happened while scoring the revision-id, please "
                    "contact the ML-Team if the issue persists."
                ),
            )

    async def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        if self.run_in_process_pool:
            self.prediction_results = await self._run_in_process_pool(
                self.model.score, feature_values
            )
        else:
            self.prediction_results = self.model.score(feature_values)
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

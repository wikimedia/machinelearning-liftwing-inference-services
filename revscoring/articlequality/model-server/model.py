import asyncio
import atexit
import concurrent.futures
import logging
import os
from typing import Dict

import aiohttp
import kserve
import mwapi
from kserve import utils as kserve_utils
from revscoring import Model
from revscoring.extractors import api
from revscoring.features import trim

import events
import preprocess_utils
import extractor_utils


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
        self.CUSTOM_UA = "WMF ML team articlequality"
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self._http_client_session = None
        atexit.register(self._shutdown)
        # The default thread pool executor set by Kserve in [1] is meant
        # for blocking I/O calls. In our cose we run async HTTP calls only,
        # and we need separate processes to run blocking CPU-bound code
        # to score revision ids.
        # [1]: https://github.com/kserve/kserve/blob/release-0.8/python/kserve/kserve/model_server.py#L129-L130
        asyncio_aux_workers = os.environ.get("ASYNCIO_AUX_WORKERS")
        if asyncio_aux_workers is None:
            asyncio_aux_workers = min(32, kserve_utils.cpu_count() + 4)
        else:
            asyncio_aux_workers = int(asyncio_aux_workers)

        logging.info(
            "Create a process pool of {} workers to support "
            "model scoring blocking code.".format(asyncio_aux_workers)
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=asyncio_aux_workers
        )

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
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    async def preprocess(self, inputs: Dict) -> Dict:
        rev_id = preprocess_utils.get_rev_id(inputs, self.REVISION_CREATE_EVENT_KEY)
        # The predict() function needs to parse the revision_create_event
        # given as input (if any).
        self.revision_create_event = preprocess_utils.get_revision_event(
            inputs, self.REVISION_CREATE_EVENT_KEY
        )
        if self.revision_create_event:
            inputs["rev_id"] = rev_id
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")

        mw_http_cache = await extractor_utils.get_revscoring_extractor_cache(
            rev_id,
            self.CUSTOM_UA,
            self.http_client_session,
            wiki_url=wiki_url,
            wiki_host=wiki_host,
            fetch_extra_info=False,
        )

        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent=self.CUSTOM_UA),
            http_cache=mw_http_cache,
        )
        inputs[self.FEATURE_VAL_KEY] = extractor_utils.fetch_features(
            rev_id, self.model.features, self.extractor
        )
        if extended_output:
            base_feature_values = extractor_utils.fetch_features(
                rev_id, list(trim(self.model.features)), self.extractor
            )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v
                for f, v in zip(list(trim(self.model.features)), base_feature_values)
            }
        return inputs

    async def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        # The score method is blocking code,
        # so we try to offload it to the asyncio's processpool
        # executor that is sets while the model server bootstraps.
        loop = asyncio.get_event_loop()
        self.prediction_results = await loop.run_in_executor(
            self.process_pool, self.model.score, feature_values
        )
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
            # https://ores.wikimedia.org/v3/scores/enwiki/186357639/articlequality?features
            # If only rev_id is given in input.json, only the prediction results
            # will be present in the response. If the extended_output flag is true,
            # features output will be included in the response.
            output[wiki_db]["scores"][rev_id][model_name]["features"] = extended_output
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


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = ArticlequalityModel(inference_name)
    model.load()
    kserve.ModelServer(workers=1).start([model])

import os
from distutils.util import strtobool
from model_servers import RevscoringModel, RevscoringModelType
import process_utils
from concurrent.futures.process import BrokenProcessPool
from http import HTTPStatus
import tornado
import logging
import extractor_utils
import preprocess_utils
from revscoring.features import trim
from typing import Dict

class RevscoringModelMP(RevscoringModel):
    def __init__(self, name: str, model_kind: RevscoringModelType):
        super().__init__(name, model_kind)
        self.asyncio_aux_workers = int(os.environ.get("ASYNCIO_AUX_WORKERS"))
        self.preprocess_mp = strtobool(os.environ.get("PREPROCESS_MP", "True"))
        self.inference_mp = strtobool(os.environ.get("INFERENCE_MP", "True"))
        self.process_pool = process_utils.create_process_pool(
            self.asyncio_aux_workers
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

    async def score(self, feature_values):
        if self.inference_mp:
            return await self._run_in_process_pool(
                self.model.score, feature_values
        )
        else:
            return super(RevscoringModelMP, self).score(feature_values)

    async def fetch_features(self, rev_id, features, extractor, cache):
        if self.preprocess_mp:
            return await self._run_in_process_pool(
                extractor_utils.fetch_features,
                rev_id,
                features,
                extractor,
                cache,
            )
        else:
            return super(RevscoringModelMP, self).fetch_features(rev_id, features, extractor, cache)

    async def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = preprocess_utils.get_rev_id(inputs, self.REVISION_CREATE_EVENT_KEY)
        extended_output = inputs.get("extended_output", False)
        await self.set_extractor(inputs, rev_id)

        cache = {}

        # The fetch_features function can be heavily cpu-bound, it depends
        # on the complexity of the rev-id to process. Running cpu-bound
        # code inside the asyncio/tornado eventloop will block the thread
        # and cause processing delays, so we use a process pool instead
        # (still enabled/disabled as opt-in).
        # See: https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools
        inputs[self.FEATURE_VAL_KEY] = await self.fetch_features(rev_id, self.model.features, self.extractor, cache)

        if extended_output:
            bare_model_features = list(trim(self.model.features))
            base_feature_values = await self.fetch_features(rev_id, bare_model_features, self.extractor, cache)
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v for f, v in zip(bare_model_features, base_feature_values)
            }
        return inputs

    async def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        self.prediction_results = await self.score(feature_values)
        output = self.get_output(request, extended_output)
        await self.send_event()
        return output

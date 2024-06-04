import logging
import os
from concurrent.futures.process import BrokenProcessPool
from distutils.util import strtobool
from typing import Dict

from kserve.errors import InferenceError
from .model_servers import RevscoringModel, RevscoringModelType
from revscoring.features import trim

from python import process_utils
from python.preprocess_utils import validate_json_input
from revscoring_model.model_servers import extractor_utils


class RevscoringModelMP(RevscoringModel):
    def __init__(self, name: str, model_kind: RevscoringModelType):
        super().__init__(name, model_kind)
        self.asyncio_aux_workers = int(os.environ.get("ASYNCIO_AUX_WORKERS"))
        self.preprocess_mp = strtobool(os.environ.get("PREPROCESS_MP", "True"))
        self.inference_mp = strtobool(os.environ.get("INFERENCE_MP", "True"))
        self.mwapi_revid_content_threshold = int(
            os.environ.get("MWAPI_REVID_CONTENT_THRESHOLD_BYTES", "100000")
        )
        self.process_pool = process_utils.create_process_pool(self.asyncio_aux_workers)

    async def _run_in_process_pool(self, *args):
        try:
            return await process_utils.run_in_process_pool(self.process_pool, *args)
        except BrokenProcessPool:
            logging.exception("Re-creation of a newer process pool before proceeding.")
            self.process_pool = process_utils.refresh_process_pool(
                self.process_pool, self.asyncio_aux_workers
            )
            raise InferenceError(
                "An error happened while scoring the revision-id, please "
                "contact the ML-Team if the issue persists."
            )

    async def score(self, feature_values):
        if self.inference_mp:
            return await self._run_in_process_pool(self.model.score, feature_values)
        else:
            return super().score(feature_values)

    async def fetch_features(
        self, rev_id, features, extractor, cache, heavy_revid_content
    ):
        if self.preprocess_mp and heavy_revid_content:
            return await self._run_in_process_pool(
                extractor_utils.fetch_features,
                rev_id,
                features,
                extractor,
                cache,
            )
        else:
            return super().fetch_features(rev_id, features, extractor, cache, False)

    async def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        inputs = validate_json_input(inputs)
        rev_id = self.get_rev_id(inputs, self.EVENT_KEY)
        extended_output = inputs.get("extended_output", False)
        extractor = await self.get_extractor(inputs, rev_id)

        cache = {}

        # Sometimes the MWAPI returns a big JSON payload for a specific rev-id
        # content, that will likely require a lot more computation for parsing
        # and tokenization. Inspect the size reported by the MWAPI-Cache and
        # offload the fetch_features call to a separate process only if it
        # crosses a certain threshold, as performance compromise to make sure
        # that the latency penalty for serialization/deserialization is not paid
        # for small/quick rev-id contents.
        heavy_revid_content = False
        revisions = extractor.http_cache.get_revisions_batch_doc([rev_id])
        for page_id, page_info in revisions["query"]["pages"].items():
            for cached_revid_result in page_info["revisions"]:
                try:
                    if (
                        cached_revid_result["size"]
                        >= self.mwapi_revid_content_threshold
                    ):
                        heavy_revid_content = True
                        logging.info(
                            f"The MWAPI cache content for {rev_id} is {cached_revid_result['size']} "
                            f"bytes and multiprocessing is used for extracting features"
                        )
                except KeyError:
                    logging.error(
                        f"The MWAPI cache content for {rev_id} didn't contain a size field, "
                        "assuming a non heavy rev-id."
                    )
                    logging.debug(
                        f"The MWAPI cached content that led to the KeyError is {cached_revid_result}",
                        cached_revid_result,
                    )

        # The fetch_features function can be heavily cpu-bound, it depends
        # on the complexity of the rev-id to process. Running cpu-bound
        # code inside the asyncio eventloop will block the thread
        # and cause processing delays, so we use a process pool instead
        # (still enabled/disabled as opt-in).
        # See: https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools
        inputs[self.FEATURE_VAL_KEY] = await self.fetch_features(
            rev_id, self.model.features, extractor, cache, heavy_revid_content
        )

        if extended_output:
            bare_model_features = list(trim(self.model.features))
            base_feature_values = await self.fetch_features(
                rev_id, bare_model_features, extractor, cache, heavy_revid_content
            )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v for f, v in zip(bare_model_features, base_feature_values)
            }
        return inputs

    async def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        self.prediction_results = await self.score(feature_values)
        output = self.get_output(request, extended_output)
        await self.send_event()
        return output

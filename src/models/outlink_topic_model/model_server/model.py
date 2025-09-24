import logging
import os
from http import HTTPStatus

import aiohttp
import fasttext
import kserve
import mwapi
from fastapi import HTTPException
from kserve.errors import InferenceError, InvalidInput

from python import events
from python.logging_utils import set_log_level
from python.preprocess_utils import (
    get_lang,
    get_page_id,
    is_domain_wikipedia,
    validate_json_input,
)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class OutlinksTopicModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

        self.EVENT_KEY = "event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        self.WIKI_URL = os.environ.get("WIKI_URL")

        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self.CUSTOM_UA = "WMF ML Team outlink-topic-model svc"
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self._http_client_session = {}

        set_log_level()
        self.MODEL_VERSION = os.environ.get("MODEL_VERSION")
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.bin")
        self.load()
        self.ready = True

    def load(self):
        self.model = fasttext.load_model(self.model_path)

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

    async def send_event(self) -> None:
        # Send a topic_prediction event to EventGate, generated from
        # the page_change event passed as input.
        topic_prediction_event = events.generate_prediction_classification_event(
            self.page_change_event,
            self.EVENTGATE_STREAM,
            "outlink-topic-model",
            self.MODEL_VERSION,
            self.prediction_results,
        )
        await events.send_event(
            topic_prediction_event,
            self.EVENTGATE_URL,
            self.TLS_CERT_BUNDLE_PATH,
            self.CUSTOM_UA,
            self.get_http_client_session("eventgate"),
        )

    async def get_outlinks(self, page_id: int, lang: str, limit=1000) -> set:
        """Gather set of up to `limit` outlinks for an article."""
        session = mwapi.AsyncSession(
            host=self.WIKI_URL or f"https://{lang}.wikipedia.org",
            user_agent="WMF ML Team outlink-topic-model svc",
            session=self.get_http_client_session("mwapi"),
        )
        session.headers["Host"] = f"{lang}.wikipedia.org"
        # generate list of all outlinks (to namespace 0) from
        # the article and their associated Wikidata IDs
        result = await session.get(
            action="query",
            generator="links",
            redirects="",
            prop="pageprops",
            ppprop="wikibase_item",
            gplnamespace=0,
            gpllimit=500,
            format="json",
            formatversion=2,
            continuation=True,
            pageids=page_id,
        )
        outlink_qids = set()
        try:
            async for r in result:
                for outlink in r["query"]["pages"]:
                    # namespace 0 and not a red link
                    if outlink["ns"] == 0 and "missing" not in outlink:
                        qid = outlink.get("pageprops", {}).get("wikibase_item", None)
                        if qid is not None:
                            outlink_qids.add(qid)
                if len(outlink_qids) > limit:
                    break
        except KeyError as e:
            logging.warning("%r Page ID: %s Lang: %s", e, str(page_id), lang)
            logging.warning("MW API returned: %r", r)
            if hasattr(self, "source_event"):
                logging.warning("Logging source event: %s", self.source_event)
        logging.debug(
            "%s (%s) fetched %d outlinks", str(page_id), lang, len(outlink_qids)
        )
        return outlink_qids

    async def _get_page_id_from_page_title(self, page_title: str, lang: str) -> int:
        session = mwapi.AsyncSession(
            host=self.WIKI_URL or f"https://{lang}.wikipedia.org",
            user_agent="WMF ML Team outlink-topic-model svc",
            session=self.get_http_client_session("mwapi"),
        )
        session.headers["Host"] = f"{lang}.wikipedia.org"
        result = await session.get(action="query", titles=page_title)
        try:
            page_id = list(result["query"]["pages"].keys())[0]
        except KeyError as e:
            logging.error("Could not find `page_id` for title: '%s'", page_title)
            logging.error("%s", str(e))
            raise
        return int(page_id)

    async def retrieve_page_id_and_title(self, inputs: dict, lang: str) -> tuple:
        # Case 1: If we are processing an event, use the `page_id` by default
        if self.EVENT_KEY in inputs:
            page_id = get_page_id(inputs, self.EVENT_KEY)
            page_title = None  # Process based on ID for events
        # Case 2: User passed both title and ID, throw an error
        elif inputs.get("page_title") and inputs.get("page_id"):
            raise InvalidInput(
                "Detected both `page_title` and `page_id` in the request. "
                "Please pass only one of those."
            )
        # Case 3: User passed title, get ID from MW API
        elif inputs.get("page_title"):
            page_title = inputs.get("page_title")
            page_id = await self._get_page_id_from_page_title(page_title, lang)
        # Case 4: User passed ID, continue with ID
        elif inputs.get("page_id"):
            page_id = inputs.get("page_id")
            page_title = None
        # Case 5: Neither title nor ID detected, throw an error
        else:
            raise InvalidInput("You must pass either `page_title` or `page_id`.")

        return (page_id, page_title)

    async def preprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        inputs = validate_json_input(inputs)
        lang = get_lang(inputs, self.EVENT_KEY)
        page_id, page_title = await self.retrieve_page_id_and_title(inputs, lang=lang)
        threshold = inputs.get("threshold", 0.5)
        if not isinstance(threshold, float):
            logging.error("Expected threshold to be a float")
            raise InvalidInput('Expected "threshold" to be a float')
        debug = inputs.get("debug", False)
        if debug:
            # when debug is enabled, we want to return all the
            # predicted topics, so it sets the threshold to 0
            debug = True
            threshold = 0.0
        if self.EVENT_KEY in inputs:
            self.source_event = inputs[self.EVENT_KEY]
            if not is_domain_wikipedia(self.source_event):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=(
                        "This model is not recommended for use in projects outside of Wikipedia"
                        " — e.g. Wiktionary, Wikinews, etc."
                    ),
                )
        if "features_str" in inputs:
            # If the features are already provided in the input,
            # we don't need to call the MW API to get features
            features_str = inputs["features_str"]
        else:
            try:
                outlinks = await self.get_outlinks(page_id=page_id, lang=lang)
            except Exception:
                if self.EVENT_KEY in inputs:
                    logging.info("Logging source event: %s", inputs[self.EVENT_KEY])
                logging.exception(
                    "Unexpected error while trying to get outlinks from MW API"
                )
                raise InferenceError(
                    "An unexpected error has occurred while trying "
                    "to get outlinks from MW API. "
                    "Please contact the ML team for more info."
                )
            features_str = " ".join(outlinks)
        request = {
            "features_str": features_str,
            "page_id": page_id,
            "page_title": page_title,
            "lang": lang,
            "threshold": threshold,
            "debug": debug,
        }
        if self.EVENT_KEY in inputs:
            request[self.EVENT_KEY] = inputs[self.EVENT_KEY]
        return request

    def postprocess(self, outputs: dict, headers: dict[str, str] = None) -> dict:
        topics = outputs["topics"]
        lang = outputs["lang"]
        page_id = outputs["page_id"]
        page_title = outputs["page_title"]
        if page_title is not None:
            article_url = f"https://{lang}.wikipedia.org/wiki/{page_title}"
        else:
            article_url = f"https://{lang}.wikipedia.org/wiki?curid={page_id}"
        result = {
            "article": article_url,
            "results": [{"topic": t[0], "score": t[1]} for t in topics],
        }
        return {"prediction": result}

    async def predict(self, request: dict, headers: dict[str, str] = None) -> dict:
        features_str = request["features_str"]
        page_id = request["page_id"]
        page_title = request["page_title"]
        lang = request["lang"]
        threshold = request["threshold"]
        debug = request["debug"]
        lbls, scores = self.model.predict(features_str, k=-1)
        results = {lb: s for lb, s in zip(lbls, scores)}
        sorted_res = [
            (lb.replace("__label__", ""), results[lb])
            for lb in sorted(results, key=results.get, reverse=True)
        ]
        above_threshold = [r for r in sorted_res if r[1] >= threshold]
        lbls_above_threshold = []
        if above_threshold:
            for res in above_threshold:
                if debug:
                    logging.info("{}: {:.3f}".format(*res))
                if res[1] > threshold:
                    lbls_above_threshold.append(res[0])
        elif debug:
            logging.info(f"No label above {threshold} threshold.")
            logging.info(
                "Top result: {} ({:.3f}) -- {}".format(
                    sorted_res[0][0], sorted_res[0][1], sorted_res[0][2]
                )
            )
        if self.EVENT_KEY in request:
            self.prediction_results = {
                "predictions": lbls_above_threshold,
                "probabilities": {r[0]: r[1] for r in sorted_res},
            }
            self.page_change_event = request[self.EVENT_KEY]
            await self.send_event()
        return {
            "topics": above_threshold,
            "lang": lang,
            "page_id": page_id,
            "page_title": page_title,
        }


if __name__ == "__main__":
    model = OutlinksTopicModel("outlink-topic-model")
    kserve.ModelServer(workers=1).start([model])

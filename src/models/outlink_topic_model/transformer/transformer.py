import argparse
import logging
import os
from http import HTTPStatus
from typing import Dict, Set

import aiohttp
import kserve
import mwapi
from fastapi import HTTPException
from kserve.errors import InferenceError, InvalidInput
from python.logging_utils import set_log_level
from python.preprocess_utils import (
    get_lang,
    get_page_title,
    is_domain_wikipedia,
    validate_json_input,
)


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class OutlinkTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.EVENT_KEY = "event"
        self.WIKI_URL = os.environ.get("WIKI_URL")
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self._http_client_session = {}
        # FIXME: this may not be needed, in theory we could simply rely on
        # kserve.constants.KSERVE_LOGLEVEL (passing KSERVE_LOGLEVEL as env var)
        # but it doesn't seem to work.
        set_log_level()
        self.ready = True

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

    async def get_outlinks(self, title: str, lang: str, limit=1000) -> Set:
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
            titles=title,
            redirects="",
            prop="pageprops",
            ppprop="wikibase_item",
            gplnamespace=0,
            gpllimit=500,
            format="json",
            formatversion=2,
            continuation=True,
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
            logging.warning("%r Title: %s Lang: %s", e, title, lang)
            logging.warning("MW API returned: %r", r)
            if hasattr(self, "source_event"):
                logging.warning("Logging source event: %s", self.source_event)
        logging.debug("%s (%s) fetched %d outlinks", title, lang, len(outlink_qids))
        return outlink_qids

    async def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = validate_json_input(inputs)
        lang = get_lang(inputs, self.EVENT_KEY)
        page_title = get_page_title(inputs, self.EVENT_KEY)
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
                outlinks = await self.get_outlinks(page_title, lang)
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
            "page_title": page_title,
            "lang": lang,
            "threshold": threshold,
            "debug": debug,
        }
        if self.EVENT_KEY in inputs:
            request[self.EVENT_KEY] = inputs[self.EVENT_KEY]
        return request

    def postprocess(self, outputs: Dict, headers: Dict[str, str] = None) -> Dict:
        topics = outputs["topics"]
        lang = outputs["lang"]
        page_title = outputs["page_title"]
        result = {
            "article": f"https://{lang}.wikipedia.org/wiki/{page_title}",
            "results": [{"topic": t[0], "score": t[1]} for t in topics],
        }
        return {"prediction": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

    args, _ = parser.parse_known_args()

    transformer = OutlinkTransformer(
        args.model_name, predictor_host=args.predictor_host
    )
    kserver = kserve.ModelServer()
    kserver.start(models=[transformer])

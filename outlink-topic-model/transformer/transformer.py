import os
import logging
import asyncio
import atexit
import argparse
from typing import Dict, Set
from http import HTTPStatus

import kserve
import mwapi
import aiohttp

import logging_utils
import preprocess_utils

from tornado.web import HTTPError

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class OutlinkTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.WIKI_URL = os.environ.get("WIKI_URL")
        self._http_client_session = aiohttp.ClientSession()
        atexit.register(self._shutdown)
        # FIXME: this may not be needed, in theory we could simply rely on
        # kserve.constants.KSERVE_LOGLEVEL (passing KSERVE_LOGLEVEL as env var)
        # but it doesn't seem to work.
        logging_utils.set_log_level()

    @property
    def http_client_session(self):
        if self._http_client_session.closed:
            logging.info("Asyncio session closed, opening a new one.")
            self._http_client_session = aiohttp.ClientSession()
        return self._http_client_session

    def _shutdown(self):
        if not self._http_client_session.closed:
            logging.info("Closing asyncio session")
            asyncio.run(self._http_client_session.close())

    async def get_outlinks(self, title: str, lang: str, limit=1000) -> Set:
        """Gather set of up to `limit` outlinks for an article."""
        session = mwapi.AsyncSession(
            host=self.WIKI_URL or f"https://{lang}.wikipedia.org",
            user_agent="WMF ML Team outlink-topic-model svc",
            session=self.http_client_session,
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
            logging.error("KeyError occurs for %s (%s). Reason: %r.", title, lang, e)
            logging.error("MW API returns: %r", r)
        logging.debug("%s (%s) fetched %d outlinks", title, lang, len(outlink_qids))
        return outlink_qids

    async def preprocess(self, inputs: Dict) -> Dict:
        lang = preprocess_utils.get_lang(inputs, self.REVISION_CREATE_EVENT_KEY)
        page_title = preprocess_utils.get_page_title(
            inputs, self.REVISION_CREATE_EVENT_KEY
        )
        threshold = inputs.get("threshold", 0.5)
        if not isinstance(threshold, float):
            logging.error("Expected threshold to be a float")
            raise HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "threshold" to be a float',
            )
        debug = inputs.get("debug", False)
        if debug:
            # when debug is enabled, we want to return all the
            # predicted topics, so it sets the threshold to 0
            debug = True
            threshold = 0.0
        if "features_str" in inputs:
            features_str = inputs["features_str"]
        else:
            try:
                outlinks = await self.get_outlinks(page_title, lang)
            except Exception:
                logging.exception(
                    "Unexpected error while trying to get outlinks from MW API"
                )
                raise HTTPError(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    reason=(
                        "An unexpected error has occurred while trying "
                        "to get outlinks from MW API. "
                        "Please contact the ML team for more info."
                    ),
                )
            features_str = " ".join(outlinks)
        request = {
            "features_str": features_str,
            "page_title": page_title,
            "lang": lang,
            "threshold": threshold,
            "debug": debug,
        }
        if self.REVISION_CREATE_EVENT_KEY in inputs:
            request[self.REVISION_CREATE_EVENT_KEY] = inputs[
                self.REVISION_CREATE_EVENT_KEY
            ]
        return request

    def postprocess(self, outputs: Dict) -> Dict:
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
    parser.add_argument(
        "--model_name",
        help="The name that the model is served under.",
    )
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )

    args, _ = parser.parse_known_args()

    transformer = OutlinkTransformer(
        args.model_name, predictor_host=args.predictor_host
    )
    kserver = kserve.ModelServer()
    kserver.start(models=[transformer])

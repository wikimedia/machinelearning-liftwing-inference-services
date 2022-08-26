import os
import logging
import asyncio
import atexit
from typing import Dict, Set
from http import HTTPStatus

import kserve
import mwapi
import tornado.web
import aiohttp

import preprocess_utils

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class OutlinkTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.CUSTOM_UA = "WMF ML Team outlink-topic-model svc"
        self._http_client_session = aiohttp.ClientSession()
        atexit.register(self._shutdown)

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
        wiki_url = os.environ.get("WIKI_URL")
        if wiki_url is None:
            # other domains like wikibooks etc are not supported.
            wiki_url = "https://{0}.wikipedia.org".format(lang)
        if wiki_url.endswith("wmnet"):
            # accessing MediaWiki API from within internal networks
            # is to use https://api-ro.discovery.wmnet and set the
            # HTTP Host header to the domain of the site you want
            # to access.
            self.http_client_session.headers.update(
                {"Host": "{0}.wikipedia.org".format(lang)}
            )
        session = mwapi.AsyncSession(
            wiki_url,
            user_agent=self.CUSTOM_UA,
            session=self.http_client_session,
        )
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
        async for r in result:
            for outlink in r["query"]["pages"]:
                # namespace 0 and not a red link
                if outlink["ns"] == 0 and "missing" not in outlink:
                    qid = outlink.get("pageprops", {}).get("wikibase_item", None)
                    if qid is not None:
                        outlink_qids.add(qid)
            if len(outlink_qids) > limit:
                break
        return outlink_qids

    async def preprocess(self, inputs: Dict) -> Dict:
        lang = preprocess_utils.get_lang(inputs, self.REVISION_CREATE_EVENT_KEY)
        page_title = preprocess_utils.get_page_title(
            inputs, self.REVISION_CREATE_EVENT_KEY
        )
        threshold = inputs.get("threshold", 0.5)
        if not isinstance(threshold, float):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Threshold value provided not a float",
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
            except KeyError:
                # No matching article or the page has no outlinks
                raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="No matching article or the page has no outlinks",
                )
            except Exception as e:
                logging.error(
                    "Unexpected error while trying to get outlinks "
                    "from MW API: {}".format(e)
                )
                raise tornado.web.HTTPError(
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
            "article": "https://{0}.wikipedia.org/wiki/{1}".format(lang, page_title),
            "results": [{"topic": t[0], "score": t[1]} for t in topics],
        }
        return {"prediction": result}

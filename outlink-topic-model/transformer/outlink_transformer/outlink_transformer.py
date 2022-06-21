import kserve
from typing import Dict, Set
import logging
import mwapi
import os
import tornado.web
from http import HTTPStatus
import asyncio
import aiohttp

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


async def get_outlinks(title: str, lang: str, limit=1000) -> Set:
    """Gather set of up to `limit` outlinks for an article."""
    async with aiohttp.ClientSession() as s:
        session = mwapi.AsyncSession(
            "https://{0}.wikipedia.org".format(lang),
            user_agent=os.environ.get("CUSTOM_UA"),
            session=s,
        )
        # generate list of all outlinks (to namespace 0) from
        # the article and their associated Wikidata IDs
        result = await asyncio.create_task(
            session.get(
                action="query",
                generator="links",
                titles=title,
                redirects="",
                prop="pageprops",
                ppprop="wikibase_item",
                gplnamespace=0,
                gpllimit=50,
                format="json",
                formatversion=2,
                continuation=True,
            )
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


class OutlinkTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    async def preprocess(self, inputs: Dict) -> Dict:
        """Get outlinks and features_str. Returns dict."""
        try:
            lang = inputs["lang"]
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Missing "lang" in input data.',
            )
        try:
            page_title = inputs["page_title"]
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Missing "page_title" in input data.',
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
        try:
            outlinks = await get_outlinks(page_title, lang)
        except KeyError:
            # No matching article or the page has no outlinks
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="No matching article or the page has no outlinks",
            )
        except RuntimeError:
            logging.error(
                "MediaWiki returned an error."
                " lang - {}, title - {}".format(lang, page_title),
                exc_info=True,
            )
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason="An internal error encountered. Please contact the WMF ML team.",
            )
        features_str = " ".join(outlinks)
        return {
            "features_str": features_str,
            "page_title": page_title,
            "lang": lang,
            "threshold": threshold,
            "debug": debug,
        }

    def postprocess(self, outputs: Dict) -> Dict:
        topics = outputs["topics"]
        lang = outputs["lang"]
        page_title = outputs["page_title"]
        result = {
            "article": "https://{0}.wikipedia.org/wiki/{1}".format(lang, page_title),
            "results": [{"topic": t[0], "score": t[1]} for t in topics],
        }
        return {"prediction": result}

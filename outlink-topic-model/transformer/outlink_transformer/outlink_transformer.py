import kserve
from typing import Dict, Set
import logging
import mwapi
import os
import tornado.web
from http import HTTPStatus

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


def get_outlinks(title: str, lang: str, limit=1000, session=None) -> Set:
    """Gather set of up to `limit` outlinks for an article."""
    if session is None:
        session = mwapi.Session(
            "https://{0}.wikipedia.org".format(lang),
            user_agent=os.environ.get("CUSTOM_UA"),
        )

    # generate list of all outlinks (to namespace 0) from
    # the article and their associated Wikidata IDs
    result = session.get(
        action="query",
        generator="links",
        titles=title,
        redirects="",
        prop="pageprops",
        ppprop="wikibase_item",
        gplnamespace=0,  # this actually doesn't seem to work :/
        gpllimit=50,
        format="json",
        formatversion=2,
        continuation=True,
    )
    try:
        outlink_qids = set()
        for r in result:
            for outlink in r["query"]["pages"]:
                # namespace 0 and not a red link
                if outlink["ns"] == 0 and "missing" not in outlink:
                    qid = outlink.get("pageprops", {}).get("wikibase_item", None)
                    if qid is not None:
                        outlink_qids.add(qid)
            if len(outlink_qids) > limit:
                break
        return outlink_qids
    except KeyError:
        # either the page doesn't have any outlinks or the page
        # doesn’t exist. The former case is rare according to
        # https://en.wikipedia.org/wiki/Category:All_dead-end_pages
        logging.error("Outlinks response: {}".format(r), exc_info=True)
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.BAD_REQUEST,
            reason="no matching article for https://{0}.wikipedia.org/wiki/{1}".format(
                lang, title
            ),
        )
    except Exception:
        logging.error("Outlinks response: {}".format(r), exc_info=True)
        return set()  # return empty set to join on feature_str


class OutlinkTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        """Get outlinks and features_str. Returns dict."""
        try:
            lang = inputs["lang"]
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='missing "lang" in input data.',
            )
        try:
            page_title = inputs["page_title"]
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='missing "page_title" in input data.',
            )
        threshold = inputs.get("threshold", 0.5)
        if not isinstance(threshold, float):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="threshold value provided not a float",
            )
        debug = inputs.get("debug", False)
        if debug:
            # when debug is enabled, we want to return all the
            # predicted topics, so it sets the threshold to 0
            debug = True
            threshold = 0.0
        outlinks = get_outlinks(page_title, lang)
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

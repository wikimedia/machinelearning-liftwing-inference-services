import kserve
from typing import Dict, Set
import logging
import mwapi
import os


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
    logging.info(result)
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
    except Exception:
        logging.error("Could not parse outlinks response")
        return set()  # return empty set to join on feature_str


class OutlinkTransformer(kserve.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        """Get outlinks and features_str. Returns dict."""
        lang = inputs.get("lang")
        page_title = inputs.get("page_title")
        outlinks = get_outlinks(page_title, lang)
        features_str = " ".join(outlinks)
        return {"features_str": features_str, "page_title": page_title, "lang": lang}

    def postprocess(self, outputs: Dict) -> Dict:
        topics = outputs["topics"]
        lang = outputs["lang"]
        page_title = outputs["page_title"]
        result = {
            "article": "https://{0}.wikipedia.org/wiki/{1}".format(lang, page_title),
            "results": [{"topic": t[0], "score": t[1]} for t in topics],
        }
        return {"prediction": result}

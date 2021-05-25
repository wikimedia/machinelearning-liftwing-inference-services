import kfserving
import fasttext
import os
import mwapi
from typing import Dict


class OutlinksTopicModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.model = fasttext.load_model('/mnt/models/model.bin')
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        features_str = request["features_str"]
        # outlinks = request["outlinks"]
        threshold = request.get('threshold', 0.5)
        debug = request.get('debug')
        lang = request.get('lang')
        page_title = request.get('page_title')
        lbls, scores = self.model.predict(features_str, k=-1)
        results = {lb: s for lb, s in zip(lbls, scores)}
        sorted_res = [(lb.replace("__label__", ""), results[lb])
                      for lb in sorted(results, key=results.get, reverse=True)]
        above_threshold = [r for r in sorted_res if r[1] >= threshold]
        lbls_above_threshold = []
        if above_threshold:
            for res in above_threshold:
                if debug:
                    print('{0}: {1:.3f}'.format(*res))
                if res[1] > threshold:
                    lbls_above_threshold.append(res[0])
        elif debug:
            print("No label above {0} threshold.".format(threshold))
            print("Top result: {0} ({1:.3f}) -- {2}".format(
                sorted_res[0][0], sorted_res[0][1], sorted_res[0][2]))

        return {"topics": above_threshold, "lang": lang,
                "page_title": page_title}

    def preprocess(self, inputs: Dict) -> Dict:
        """Get outlinks and features_str. Returns dict."""
        lang = inputs.get('lang')
        page_title = inputs.get('page_title')
        outlinks = self.get_outlinks(page_title, lang)
        features_str = ' '.join(outlinks)
        return {'features_str': features_str, 'page_title': page_title,
                'lang': lang}

    def postprocess(self, outputs: Dict) -> Dict:
        topics = outputs['topics']
        lang = outputs['lang']
        page_title = outputs['page_title']
        result = {'article':
                  'https://{0}.wikipedia.org/wiki/{1}'.format(
                      lang, page_title),
                  'results': [{'topic': t[0], 'score': t[1]} for t in topics]
                  }
        return {'prediction': result}

    def get_outlinks(self, title, lang, limit=1000, session=None):
        """Gather set of up to `limit` outlinks for an article."""
        if session is None:
            session = mwapi.Session(
                'https://{0}.wikipedia.org'.format(lang),
                user_agent=os.environ.get('CUSTOM_UA'))

        # generate list of all outlinks (to namespace 0) from
        # the article and their associated Wikidata IDs
        result = session.get(
            action="query",
            generator="links",
            titles=title,
            redirects='',
            prop='pageprops',
            ppprop='wikibase_item',
            gplnamespace=0,  # this actually doesn't seem to work :/
            gpllimit=50,
            format='json',
            formatversion=2,
            continuation=True
        )
        try:
            outlink_qids = set()
            for r in result:
                for outlink in r['query']['pages']:
                    # namespace 0 and not a red link
                    if outlink['ns'] == 0 and 'missing' not in outlink:
                        qid = outlink.get('pageprops', {}).get(
                            'wikibase_item', None)
                        if qid is not None:
                            outlink_qids.add(qid)
                if len(outlink_qids) > limit:
                    break
            return outlink_qids
        except Exception:
            return None


if __name__ == "__main__":
    model = OutlinksTopicModel("outlink-topic-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])

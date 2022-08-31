import os
from typing import Dict

import kserve
import fasttext


class OutlinksTopicModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.model = fasttext.load_model("/mnt/models/model.bin")
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        features_str = request["features_str"]
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
                    print("{0}: {1:.3f}".format(*res))
                if res[1] > threshold:
                    lbls_above_threshold.append(res[0])
        elif debug:
            print("No label above {0} threshold.".format(threshold))
            print(
                "Top result: {0} ({1:.3f}) -- {2}".format(
                    sorted_res[0][0], sorted_res[0][1], sorted_res[0][2]
                )
            )

        return {"topics": above_threshold, "lang": lang, "page_title": page_title}


if __name__ == "__main__":
    model = OutlinksTopicModel("outlink-topic-model")
    model.load()
    kserve.ModelServer(workers=1).start([model])

import os
import logging
from typing import Dict

import kserve
import fasttext

import events

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class OutlinksTopicModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.REVISION_CREATE_EVENT_KEY = "revision_create_event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self.CUSTOM_UA = "WMF ML Team outlink-topic-model svc"
        self.MODEL_VERSION = os.environ.get("MODEL_VERSION")
        self.load()

    def load(self):
        self.model = fasttext.load_model("/mnt/models/model.bin")
        self.ready = True

    async def predict(self, request: Dict) -> Dict:
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
        self.prediction_results = {
            "prediction": lbls_above_threshold,
            "probability": {r[0]: r[1] for r in sorted_res},
        }
        # Send a revision-score event to EventGate, generated from
        # the revision-create event passed as input.
        if self.REVISION_CREATE_EVENT_KEY in request:
            revision_score_event = events.generate_revision_score_event(
                request[self.REVISION_CREATE_EVENT_KEY],
                self.EVENTGATE_STREAM,
                self.MODEL_VERSION,
                self.prediction_results,
                "outlink",
            )
            await events.send_event(
                revision_score_event,
                self.EVENTGATE_URL,
                self.TLS_CERT_BUNDLE_PATH,
                self.CUSTOM_UA,
                self._http_client,
            )
        return {"topics": above_threshold, "lang": lang, "page_title": page_title}


if __name__ == "__main__":
    model = OutlinksTopicModel("outlink-topic-model")
    kserve.ModelServer(workers=1).start([model])

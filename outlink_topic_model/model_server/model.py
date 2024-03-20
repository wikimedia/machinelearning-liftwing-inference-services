import logging
import os
from typing import Dict

import aiohttp
import fasttext
import kserve
from python import events

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class OutlinksTopicModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.EVENT_KEY = "event"
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_STREAM = os.environ.get("EVENTGATE_STREAM")
        # Deployed via the wmf-certificates package
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self.CUSTOM_UA = "WMF ML Team outlink-topic-model svc"
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self.MODEL_VERSION = os.environ.get("MODEL_VERSION")
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.bin")
        self._http_client_session = {}
        self.load()

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

    def load(self):
        self.model = fasttext.load_model(self.model_path)
        self.ready = True

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

    async def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
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
        return {"topics": above_threshold, "lang": lang, "page_title": page_title}


if __name__ == "__main__":
    model = OutlinksTopicModel("outlink-topic-model")
    kserve.ModelServer(workers=1).start([model])

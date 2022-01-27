import kserve
import logging
import mwapi
import os
import requests
from revscoring.extractors import api
from typing import Dict


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class DrafttopicTransformer(kserve.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = inputs.get("rev_id")
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")
        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None
        extractor = api.Extractor(
            mwapi.Session(
                wiki_url, user_agent="WMF ML Team topic model", session=s
            )
        )
        feature_values = list(extractor.extract(rev_id, self.model.features))
        return {"feature_values": feature_values}

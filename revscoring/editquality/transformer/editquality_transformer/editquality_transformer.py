import kserve
import logging
import mwapi
import os
import requests
from revscoring import Model
from revscoring.extractors import api
from typing import Dict


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class EditQualityTransformer(kserve.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = False  # ensure we can load model
        self._load()

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
        self.extractor = api.Extractor(
            mwapi.Session(
                wiki_url, user_agent="WMF ML Team editquality model", session=s
            )
        )
        return self._fetch_editquality_features(rev_id)

    def _load(self):
        """Load model so we can access features."""
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    def _fetch_editquality_features(self, rev_id: int) -> Dict:
        """Retrieve editquality features."""
        feature_values = list(self.extractor.extract(rev_id, self.model.features))
        return {"feature_values": feature_values}

import kserve
import bz2
from typing import Dict
import logging
import mwapi
from revscoring import Model
from revscoring.extractors import api
import os
import requests


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class DraftQualityTransformer(kserve.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = False  # ensure we can load model
        self._load()

    def preprocess(self, inputs: Dict) -> Dict:
        """Get outlinks and features_str."""
        rev_id = inputs.get("rev_id")
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")
        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None
        ua = "WMF ML team draftquality transformer"
        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent=ua, session=s)
        )
        return self._fetch_draftquality_text(rev_id)

    def _load(self):
        """Load model so we can access features."""
        with bz2.open("/mnt/models/model.bz2") as f:
            self.model = Model.load(f)
        self.ready = True

    def _fetch_draftquality_features(self, rev_id: int) -> Dict:
        """Retrieve draftquality features."""
        feature_values = list(self.extractor.extract(rev_id, self.model.features))
        return {"feature_values": feature_values}

import logging
import os

import kserve

from model_servers import RevscoringModel, RevscoringModelType

if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = RevscoringModel(inference_name, RevscoringModelType.DRAFTQUALITY)
    kserve.ModelServer(workers=1).start([model])

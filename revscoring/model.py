import os

import kserve

from model_servers import RevscoringModel, RevscoringModelType

if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model_type = RevscoringModelType.get_model_type(inference_name)
    model = RevscoringModel(inference_name, model_type)
    kserve.ModelServer(workers=1).start([model])

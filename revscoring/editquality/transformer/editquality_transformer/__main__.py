import kserve
import argparse
from .editquality_transformer import EditQualityTransformer

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kserve.kfserver.parser])
parser.add_argument(
    "--model_name",
    default=DEFAULT_MODEL_NAME,
    help="The name that the model is served under.",
)
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = EditQualityTransformer(
        args.model_name, predictor_host=args.predictor_host
    )
    kfserver = kserve.KFServer()
    kfserver.start(models=[transformer])

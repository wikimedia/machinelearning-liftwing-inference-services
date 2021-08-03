MODEL_NAME=enwiki-goodfaith
INPUT_PATH=@./input.json
CLIENT_GENERATORS_PATH=../../../client
SESSION=$(python3 $CLIENT_GENERATORS_PATH/authservice-session-generator.py)
. $CLIENT_GENERATORS_PATH/inference-generator.sh

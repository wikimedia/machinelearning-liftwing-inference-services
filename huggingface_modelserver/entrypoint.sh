#!/bin/bash
set -e
if [ -z "$MODEL_ID" ]; then
    # Run the server from a local model directory
    python3 -m huggingfaceserver --model_dir $MODEL_DIR --model_name $MODEL_NAME
else
    # Run the server by downloading the model from HuggingFace
    python3 -m huggingfaceserver --model_id $MODEL_ID --model_name $MODEL_NAME
fi

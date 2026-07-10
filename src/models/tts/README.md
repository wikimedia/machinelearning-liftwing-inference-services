# Wikipedia Text-to-Speech (TTS)

The Wikipedia TTS inference service uses the Kokoro text-to-speech model and Wav2Vec2-CTC forced aligner in a KServe model-server that takes pre-normalized text segments and returns concatenated float32 PCM audio with word-level timestamps.

* Model Card: https://huggingface.co/hexgrad/Kokoro-82M
* Source: https://github.com/hexgrad/kokoro
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/tts/
* Model license: Apache 2.0 License

## How to run locally

In order to run the TTS model-server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using docker compose</summary>

### 1.1. Build
In the first terminal run:
```console
docker compose build tts
```

This will build a TTS image with all dependencies installed.

### 1.2. Run

On the same terminal run the model-server:
```console
docker compose up tts
```

### 1.3. Query

On the second terminal query the isvc using:
```console
curl -s -X POST http://localhost:8080/v1/models/tts:predict \
  -H 'Content-Type: application/json' \
  -d '{
    "segments": [
      {"text": "Hello world.", "voice": "af_heart"}
    ]
  }'
```

Expected response:
```console
{
    "audio_b64": "<base64-encoded PCM audio, 24 kHz mono; int16 by default>",
    "encoding": "pcm_s16le",
    "sample_rate": 24000,
    "duration_ms": 1045.3,
    "timestamps": [
        {
            "word": "Hello",
            "start_ms": 80.0,
            "end_ms": 280.0
        },
        {
            "word": "world.",
            "start_ms": 420.0,
            "end_ms": 720.0
        }
    ]
}
```

### 1.4. Remove

If you would like to remove the setup run:
```console
docker compose down -v --rmi all
```
</details>

### Request parameters

Optional top-level fields alongside `segments`:

- `encoding` — response PCM format: `pcm_s16le` (16-bit int, default) or
  `pcm_f32le` (32-bit float). int16 halves the payload and is transparent
  for speech.
- `default_voice`, `default_speed`, `default_lang` — defaults applied to
  segments that don't specify their own.

<details>
<summary>2. Manual setup</summary>

### 2.1. Build Python venv and install dependencies

First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/tts/requirements.txt
```

### 2.2. Download model files

Download the model files from the link below and place them in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/tts/
Now our PATH_TO_MODEL_DIR directory contains the model with the following structure:
```console
PATH_TO_MODEL_DIR
├── kokoro
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
└── wav2vec2
    ├── model.onnx
    └── processor
        ├── preprocessor_config.json
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── vocab.json
```

### 2.3. Run the server

We can run the server locally with:
```console
MODEL_NAME=tts MODEL_PATH=PATH_TO_MODEL_DIR python3 src/models/tts/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s -X POST http://localhost:8080/v1/models/tts:predict \
  -H 'Content-Type: application/json' \
  -d '{
    "segments": [
      {"text": "Hello world.", "voice": "af_heart"}
    ]
  }'
```

Expected response:
```console
{
    "audio_b64": "<base64-encoded PCM audio, 24 kHz mono; int16 by default>",
    "encoding": "pcm_s16le",
    "sample_rate": 24000,
    "duration_ms": 1045.3,
    "timestamps": [
        {
            "word": "Hello",
            "start_ms": 80.0,
            "end_ms": 280.0
        },
        {
            "word": "world.",
            "start_ms": 420.0,
            "end_ms": 720.0
        }
    ]
}
```
</details>

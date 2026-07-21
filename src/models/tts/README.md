# Wikipedia Text-to-Speech (TTS)

The Wikipedia TTS inference service uses the Kokoro text-to-speech model and
Wav2Vec2-CTC forced aligner in a KServe model-server that takes
pre-normalized text segments and returns concatenated PCM audio (16-bit
integer by default) with word-level timestamps.

* Model Card: https://huggingface.co/hexgrad/Kokoro-82M
* Source: https://github.com/hexgrad/kokoro
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/tts/
* Model license: Apache 2.0 License

## How to run locally

In order to run the TTS model-server locally, please choose one of the two
options below. Once the server is running, see [Input](#input) and
[Output](#output) for how to query it.

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

The service listens on `http://localhost:8080`.

### 1.3. Remove

If you would like to remove the setup run:
```console
docker compose down -v --rmi all
```
</details>

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

Download the model files from the link below and place them in the same
directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/tts/

Now our PATH_TO_MODEL_DIR directory contains the model with the following
structure:
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

The service listens on `http://localhost:8080`.
</details>

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | — | KServe model name; must be `tts` (forms the `/v1/models/tts:predict` route). |
| `MODEL_PATH` | — | Directory containing the `kokoro/` and `wav2vec2/` model files (see structure above). |
| `KOKORO_THREADS` | `2` | ONNX Runtime intra-op thread count for Kokoro synthesis. Set explicitly to the pod CPU allocation: ONNX Runtime's default (0 = one thread per host core) reads the 96-core host instead of the cgroup limit and collapses throughput (T430536). Deployment overrides to `8` for the 8-CPU staging pod. |
| `W2V2_THREADS` | `1` | ONNX Runtime intra-op thread count for the Wav2Vec2-CTC aligner. Same cgroup rationale as above. Deployment overrides to `2`. |

## Input

`POST /v1/models/tts:predict` with a JSON body. `segments` is the only
required field; each segment is synthesized in order and the audio is
concatenated into one response.

Per-segment fields:

| Field | Required | Description |
| --- | --- | --- |
| `text` | yes | Pre-normalized text to speak. Keep segments at or below 800 characters (Kokoro's practical input limit; the server warns above it and the model may silently truncate). |
| `voice` | no | Voice for this segment (falls back to `default_voice`). |
| `speed` | no | Speaking-rate multiplier (falls back to `default_speed`). |
| `lang` | no | Language code (falls back to `default_lang`). |

Optional top-level fields alongside `segments`:

| Field | Default | Description |
| --- | --- | --- |
| `encoding` | `pcm_s16le` | Response PCM format: `pcm_s16le` (16-bit int) or `pcm_f32le` (32-bit float). int16 halves the payload and is transparent for speech. |
| `timestamps` | `full` | Word-timestamp generation: `full` (CTC forced alignment), `proportional` (character-count-weighted approximation, near-zero cost), or `none` (no timestamps; fastest, audio-only). |
| `default_voice` | `af_heart` | Voice applied to segments that don't specify their own. |
| `default_speed` | `1.0` | Speed applied to segments that don't specify their own. |
| `default_lang` | `en-us` | Language applied to segments that don't specify their own. |

Example request:
```console
curl -s -X POST http://localhost:8080/v1/models/tts:predict \
  -H 'Content-Type: application/json' \
  -d '{
    "segments": [
      {"text": "Hello world.", "voice": "af_heart"}
    ]
  }'
```

## Output

A JSON body with the concatenated audio and (unless `timestamps: none`)
one timestamp entry per spoken word:

| Field | Description |
| --- | --- |
| `audio_b64` | Base64-encoded PCM audio, 24 kHz mono, in the requested `encoding`. |
| `encoding` | PCM format of `audio_b64` (`pcm_s16le` or `pcm_f32le`). |
| `timestamps_mode` | The timestamp mode that was applied (`full`, `proportional`, or `none`). |
| `sample_rate` | Sample rate in Hz (always `24000`). |
| `duration_ms` | Total audio duration in milliseconds. |
| `timestamps` | Array of `{word, start_ms, end_ms}` entries across all segments; empty when `timestamps_mode` is `none`. |

Expected response for the example request above:
```console
{
    "audio_b64": "<base64-encoded PCM audio, 24 kHz mono; int16 by default>",
    "encoding": "pcm_s16le",
    "timestamps_mode": "full",
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

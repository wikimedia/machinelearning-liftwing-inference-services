
### Running locally

Let's say we want to run a LLM model server locally. First we need to create a virtual environment and install the dependencies:

This can be done with:
```console
python -m venv .venv
source .venv/bin/activate
pip install -r llm/requirements.txt
```

To be able to use utils from the python director, we need to add the top level directory to the PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:<PATH_TO_INFERENCE_SERVICES_REPO>

Then running:
>  MODEL_NAME=nllb-200 LLM_CLASS=llm.NLLB MODEL_PATH=/path/to/model/files/ python llm/model.py

Make a request:
> curl localhost:8080/v1/models/nllb-200:predict -i -X POST -d '{"prompt": "Some random text we want to translate to german", "tgt_lang": "deu_Latn"}'

### Using ctranslate2
For some models we can use a special runtime called ctranslate2. This runtime is much faster than the default runtime
and is available only for some models. To use it, you need to install ctranslate2 and its dependencies:

> pip install ctranslate2

Then you can convert the model to ctranslate2 format using the following command:
> ct2-transformers-converter --model facebook/nllb-200-distilled-600M --quantization int8 --output_dir /path/to/output/dir

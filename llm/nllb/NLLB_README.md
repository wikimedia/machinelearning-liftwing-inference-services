# NLLB
We are using the 600M parameter model available by facebook under a CC BY-NC 4.0 license.
The original version of the model can be found on [Hugging Face](https://huggingface.co/facebook/nllb-200-distilled-600M).

### Running locally
Let's say we want to run nllb model server locally. First we need to create a virtual environment and install the dependencies:

This can be done with:
```console
python -m venv .venv
source .venv/bin/activate
pip install -r llm/requirements.txt
```
To be able to use utils from the python director, we need to add the top level directory to the PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:<PATH_TO_INFERENCE_SERVICES_REPO>
>
Then running:
>  MODEL_NAME=nllb-200 LLM_CLASS=llm.NLLB MODEL_PATH=/path/to/model/files/ python llm/model.py
>
Make a request:
> curl localhost:8080/v1/models/nllb-200:predict -i -X POST -d '{"prompt": "Some random text we want to translate to german", "tgt_lang": "deu_Latn"}'

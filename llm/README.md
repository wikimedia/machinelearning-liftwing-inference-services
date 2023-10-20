
### Running locally
If you want to run the model servers locally you can do so by first adding the top level dir to the PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:
>
Then running:
>  MODEL_NAME=nllb-200 LLM_CLASS=nllb.NLLB MODEL_PATH=/path/to/model/binary/ python llm/model-server/model.py
>
Make a request:
> curl localhost:8080/v1/models/nllb-200:predict -i -X POST -d '{"prompt": "Some random text we want to translate to german", "tgt_lang": "deu_Latn"}'

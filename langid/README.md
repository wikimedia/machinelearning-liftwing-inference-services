# Language Identification

Language identification service supporting 201 languages.

* Source: https://github.com/laurieburchell/open-lid-dataset
* Paper: https://arxiv.org/pdf/2305.13820.pdf
* Model: https://data.statmt.org/lid/lid201-model.bin.gz and
 https://analytics.wikimedia.org/published/wmf-ml-models/langid/
* Model license: the GNU General Public License v3.0.



### Running locally
If you want to run the model servers locally you can do so by first adding the top level dir to the PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:
>
Then running:
>  MODEL_NAME=langid MODEL_PATH=/path/to/model/binary/lid201-model.bin python langid/model.py
>
Make a request:
> curl localhost:8080/v1/models/langid:predict -i -X POST -d '{"text": "Some random text in any language"}'

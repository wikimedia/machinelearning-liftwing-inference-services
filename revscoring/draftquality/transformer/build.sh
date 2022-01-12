# Installs the NLTK stopwords and also installs
# the transformer module via setuptools.
# FIXME: path hack - see: https://phabricator.wikimedia.org/T267685
PYTHONPATH=/opt/lib/python/site-packages
python3 -m nltk.downloader omw sentiwordnet stopwords wordnet
python3 -m pip install --target /opt/lib/python/site-packages .

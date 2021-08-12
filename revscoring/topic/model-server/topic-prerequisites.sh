# FIXME: path hack - see: https://phabricator.wikimedia.org/T267685
PYTHONPATH=/opt/lib/python/site-packages
python3 -m nltk.downloader omw sentiwordnet stopwords wordnet

mkdir model-server/word2vec
wget -O model-server/Makefile https://raw.githubusercontent.com/wikimedia/drafttopic/master/Makefile
# download word embedding vectors from /srv/topic/model-server/Makefile by searching for strings that match the required targets.
# i.e search(grep) strings that start with "word2vec/" and remove(sed) ":" from them.
# also the first layer round brackets ( ) ensure that we don't leave the shell's current directory "/srv/topic"
(cd ./model-server && make $(grep '^word2vec/.*' Makefile | sed 's/:.$//'))
#!/bin/bash

set -ex
mkdir word2vec
wget -O Makefile https://raw.githubusercontent.com/wikimedia/drafttopic/master/Makefile
# download word embedding vectors from /srv/topic/Makefile by searching for strings that match the required targets.
# i.e search(grep) strings that start with "word2vec/" and remove(sed) ":" from them.
# also the first layer round brackets ( ) ensure that we don't leave the shell's current directory "/srv/topic"
(make $(grep '^word2vec/.*' Makefile | tr -d '[: \r]'))

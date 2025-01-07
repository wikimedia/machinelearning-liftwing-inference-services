#!/bin/bash
git init
git add .

# cater to new test variants that have tox.ini in root dir
tox -e "$1,$2"
lalalalalallaa

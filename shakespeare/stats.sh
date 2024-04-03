#!/usr/bin/env bash

NAME="shakespeare"

cd ../utils_data

python3 stats.py --name $NAME

cd ../$NAME
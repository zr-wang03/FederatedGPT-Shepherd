#!/usr/bin/env bash

# download data and convert to .json format

RAWTAG=""
if [[ $@ = *"--raw"* ]]; then
  RAWTAG="--raw"
fi
if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    sudo bash data_to_json.sh $RAWTAG
    cd ..
fi

NAME="shakespeare"

cd ../utils_data

sudo bash preprocess.sh --name $NAME $@

cd ../$NAME
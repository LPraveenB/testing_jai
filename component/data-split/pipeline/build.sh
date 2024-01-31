#!/bin/bash

build_json="${WORKSPACE}/component/build.json"
scripts_dest=$(jq -r .script_paths.data_split "$build_json")

ls ${WORKSPACE}/component/data-split/src/*.py
src_dir="${WORKSPACE}/component/data-split/"
gsutil -m cp -r "$src_dir" "$scripts_dest"
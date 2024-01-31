#!/bin/bash

build_json="${WORKSPACE}/component/build.json"
scripts_dest=$(jq -r .script_paths.switch "$build_json")

ls ${WORKSPACE}/component/switch/src/*.py
src_dir="${WORKSPACE}/component/switch/*"
gsutil -m cp -r "$src_dir" "$scripts_dest"
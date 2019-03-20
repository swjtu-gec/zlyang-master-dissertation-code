#!/usr/bin/env bash

commit=ec5c7b009c409e72b5ef65a77c1a846546f14847
echo "Downloading Subword NMT from https://github.com/rsennrich/subword-nmt (rev: ${commit})"
wget https://github.com/rsennrich/subword-nmt/archive/${commit}.zip
unzip ${commit}.zip
rm ${commit}.zip
mv subword-nmt-${commit} subword-nmt

commit=cb874a7ecdcb360b08635538a299cf868258a4c1
echo "Downloading m2scorer from https://github.com/nusnlp/m2scorer (rev: ${commit})"
wget https://github.com/nusnlp/m2scorer/archive/${commit}.zip
unzip ${commit}.zip
rm ${commit}.zip
mv m2scorer-${commit} m2scorer


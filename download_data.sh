#!/usr/bin/env bash
set -euo pipefail

URL="https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
ARCHIVE="dataset-resized.zip"

echo "Downloading trashnet dataset..."
curl -L -o "$ARCHIVE" "$URL"

echo "Extracting..."
unzip -q "$ARCHIVE" -d data_tmp

rm -rf data/trashnet && mkdir -p data
mv data_tmp/dataset-resized data/trashnet
rm -rf data_tmp "$ARCHIVE"
find data/trashnet -name ".DS_Store" -delete

echo "Done. Data available in ./data/trashnet/"

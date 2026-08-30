#!/usr/bin/env bash
# Download the public Kaggle Cats-vs-Dogs dataset into data/raw/{cat,dog}.
#
# Kaggle made public dataset downloads available through its API without
# authentication. This keeps the same `make data-kaggle` command usable both
# locally and in GitHub Actions without storing a Kaggle username or API key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
TMP="$ROOT/data/.kaggle-tmp"
ARCHIVE="$TMP/catsvsdogs.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/shaunthesheep/microsoft-catsvsdogs-dataset"

mkdir -p "$TMP" "$RAW/cat" "$RAW/dog"
rm -rf "$RAW/cat"/* "$RAW/dog"/*

echo "downloading public Kaggle dataset..."
curl -fL --retry 3 --retry-delay 2 "$DATASET_URL" -o "$ARCHIVE"
unzip -q -o "$ARCHIVE" -d "$TMP/extracted"

find "$TMP/extracted" -type d -iname 'cat' -exec sh -c 'cp "$1"/*.jpg "$2"/ 2>/dev/null || true' _ {} "$RAW/cat" \;
find "$TMP/extracted" -type d -iname 'dog' -exec sh -c 'cp "$1"/*.jpg "$2"/ 2>/dev/null || true' _ {} "$RAW/dog" \;

rm -rf "$TMP"

CAT_COUNT=$(find "$RAW/cat" -type f | wc -l | tr -d ' ')
DOG_COUNT=$(find "$RAW/dog" -type f | wc -l | tr -d ' ')

test "$CAT_COUNT" -gt 0 || { echo "No cat images were downloaded" >&2; exit 1; }
test "$DOG_COUNT" -gt 0 || { echo "No dog images were downloaded" >&2; exit 1; }

echo "cat images: $CAT_COUNT"
echo "dog images: $DOG_COUNT"
echo "next: make preprocess"

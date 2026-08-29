#!/usr/bin/env bash
# Download the real Kaggle cats-vs-dogs archive into data/raw/{cat,dog}.
#
# Requires the kaggle CLI and ~/.kaggle/kaggle.json. If you do not have
# credentials, `make data` generates an equivalent synthetic dataset instead and
# every later stage behaves identically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
TMP="$ROOT/data/.kaggle-tmp"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with: pip install kaggle" >&2
  echo "Then place your token at ~/.kaggle/kaggle.json (chmod 600)." >&2
  exit 1
fi

mkdir -p "$TMP" "$RAW/cat" "$RAW/dog"
echo "downloading microsoft/catsvsdogs…"
kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p "$TMP" --unzip

find "$TMP" -type d -iname 'cat' -exec sh -c 'cp "$1"/*.jpg "$2"/ 2>/dev/null || true' _ {} "$RAW/cat" \;
find "$TMP" -type d -iname 'dog' -exec sh -c 'cp "$1"/*.jpg "$2"/ 2>/dev/null || true' _ {} "$RAW/dog" \;
rm -rf "$TMP"

echo "cat images: $(find "$RAW/cat" -type f | wc -l)"
echo "dog images: $(find "$RAW/dog" -type f | wc -l)"
echo "next: make preprocess"

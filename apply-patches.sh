#!/usr/bin/env bash
# Apply the three Swiss AI vLLM patches that EAGLE-3 needs on Apertus 1.5.
# usage: apply-patches.sh <vllm-source-dir>   (the directory that contains vllm/, e.g. /workspace/vllm)
# The patches are fetched from the Swiss AI repository at a pinned commit and checked against a sha256.
set -euo pipefail
SRC="${1:?usage: $0 <vllm-source-dir>}"
REPO=swiss-ai/apertus-speculative-decoding
COMMIT=87a3f19e07adaff2a68deb4eb517fb6507f20820
declare -A SUM=(
  [vllm-apertus-eagle3-aux-layers.patch]=78938e9fb90a722ad420eddfd009b7ca8992d3299bd025dacf06688b8c02c043
  [vllm-apertus-eagle-mrv2-inner.patch]=626495bbdbde9c7b7dac28440b503cfcbba10cbf90f9daddc78b9bf5b0b453c9
  [vllm-apertus-image-token.patch]=3f4e41d37cfa187836229aafcd628766671de19ed89c006369ba7ae390587bc5
)
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
for p in vllm-apertus-eagle3-aux-layers.patch vllm-apertus-eagle-mrv2-inner.patch vllm-apertus-image-token.patch; do
  curl -fsSL -o "$tmp/$p" "https://raw.githubusercontent.com/$REPO/$COMMIT/serving/patches/$p"
  echo "${SUM[$p]}  $tmp/$p" | sha256sum -c -
  patch -p3 -d "$SRC" --dry-run < "$tmp/$p" >/dev/null   # refuse to half-apply
done
for p in vllm-apertus-eagle3-aux-layers.patch vllm-apertus-eagle-mrv2-inner.patch vllm-apertus-image-token.patch; do
  patch -p3 -d "$SRC" < "$tmp/$p"
done

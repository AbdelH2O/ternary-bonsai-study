#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
src_dir=$(mktemp -d)
trap 'rm -rf "$src_dir"' EXIT
archive="$src_dir/prism.tar.gz"
curl -L --fail --silent --show-error https://github.com/PrismML-Eng/llama.cpp/archive/842b188.tar.gz -o "$archive"
printf '%s  %s\n' 84ec38b7e7fb45a9f076e923e064e967e7c30b946b71b3b778444b05e8459c3c "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$src_dir" --strip-components=1
g++ -std=c++17 -O2 -I"$src_dir/include" -I"$src_dir/ggml/include" \
  "$repo_root/analysis/bonsai2/scorer/position_scorer.cpp" \
  -L"$repo_root/bin/cuda" -Wl,-rpath,"$repo_root/bin/cuda" -lllama \
  -o "$repo_root/analysis/bonsai2/scorer/position_scorer"

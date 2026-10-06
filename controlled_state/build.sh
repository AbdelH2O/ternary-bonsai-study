#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
source_dir=$(mktemp -d)
trap 'rm -rf "$source_dir"' EXIT
archive="$source_dir/prism.tar.gz"
curl -L --fail --silent --show-error https://github.com/PrismML-Eng/llama.cpp/archive/842b188.tar.gz -o "$archive"
printf '%s  %s\n' 84ec38b7e7fb45a9f076e923e064e967e7c30b946b71b3b778444b05e8459c3c "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$source_dir" --strip-components=1
g++ -std=c++17 -O2 -I"$source_dir/include" -I"$source_dir/ggml/include" \
  "$root/analysis/bonsai2/controlled_state/controlled_state.cpp" \
  -L"$root/bin/cuda" -Wl,-rpath,"$root/bin/cuda" -lllama \
  -o "$root/analysis/bonsai2/controlled_state/controlled_state"

#!/usr/bin/env bash
# Prove that llama.cpp/ is exactly upstream b10660 plus patches/*.patch, nothing else.
# Clones upstream into a temp dir, applies the patches with `git am`, and diffs.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone -q --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$tmp/upstream"
git -C "$tmp/upstream" checkout -q b10660
git -C "$tmp/upstream" -c user.name=verify -c user.email=verify@localhost am -q "$here"/patches/*.patch

mkdir "$tmp/expected"
git -C "$tmp/upstream" archive HEAD | tar -x -C "$tmp/expected"
if diff -r "$tmp/expected" "$here/llama.cpp" --exclude=build >/dev/null; then
    echo "OK: llama.cpp/ == b10660 + $(ls "$here"/patches/*.patch | wc -l) patches"
else
    diff -rq "$tmp/expected" "$here/llama.cpp" --exclude=build
    echo "MISMATCH: llama.cpp/ differs from b10660 + patches" >&2
    exit 1
fi

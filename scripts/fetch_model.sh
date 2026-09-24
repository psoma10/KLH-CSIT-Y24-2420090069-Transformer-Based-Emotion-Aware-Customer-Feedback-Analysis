#!/usr/bin/env bash
# Downloads the trained model artifact into MODEL_DIR.
#
# The weights are ~479MB and gitignored, so they are absent from the repo
# Render clones. This pulls them at build time instead.
#
# MODEL_URL must point at a single archive containing the contents of
# ml/artifacts/model-v1/ — config.json, model.safetensors, thresholds.json and
# the tokenizer files. Accepted formats:
#
#   .tar.gz / .tgz    tar archive, gzip compressed
#   .zip              zip archive
#
# The archive may hold those files at its top level or nested inside a single
# wrapping directory (e.g. model-v1/config.json); both layouts work.
#
# Any host that serves the archive over plain HTTPS without interactive auth is
# fine — a GitHub release asset, an S3 or R2 object, a Hugging Face resolve URL.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?MODEL_DIR is not set — render.yaml defines it per service}"
MODEL_URL="${MODEL_URL:?MODEL_URL is not set — add it in the Render dashboard (sync:false in render.yaml)}"

# Files the backend needs before it will serve /predict. Checked after
# extraction so a truncated download fails the build rather than the first
# request.
REQUIRED_FILES=(config.json model.safetensors thresholds.json tokenizer.json)

have_all_required() {
  local f
  for f in "${REQUIRED_FILES[@]}"; do
    [ -f "$MODEL_DIR/$f" ] || return 1
  done
  return 0
}

if have_all_required; then
  echo "fetch_model: artifact already present at $MODEL_DIR — skipping download"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

case "$MODEL_URL" in
  *.tar.gz|*.tgz) archive="$tmp_dir/model.tar.gz" ;;
  *.zip)          archive="$tmp_dir/model.zip" ;;
  *)
    echo "fetch_model: unsupported MODEL_URL extension." >&2
    echo "  expected .tar.gz, .tgz or .zip, got: $MODEL_URL" >&2
    exit 1
    ;;
esac

echo "fetch_model: downloading artifact (~479MB), this takes a few minutes"
# --fail turns an HTML 404 body into a non-zero exit; without it curl happily
# writes the error page to disk and the extract fails with a confusing message.
curl --fail --location --show-error --silent --output "$archive" "$MODEL_URL"

extract_dir="$tmp_dir/extracted"
mkdir -p "$extract_dir"

case "$archive" in
  *.tar.gz) tar -xzf "$archive" -C "$extract_dir" ;;
  *.zip)    unzip -q "$archive" -d "$extract_dir" ;;
esac

# Collapse a single wrapping directory, so both `model-v1/config.json` and a
# bare `config.json` layout land in the same place.
src_dir="$extract_dir"
if [ ! -f "$src_dir/config.json" ]; then
  nested="$(find "$extract_dir" -maxdepth 2 -name config.json -print -quit || true)"
  if [ -n "$nested" ]; then
    src_dir="$(dirname "$nested")"
  fi
fi

mkdir -p "$MODEL_DIR"
cp -R "$src_dir"/. "$MODEL_DIR"/

missing=()
for f in "${REQUIRED_FILES[@]}"; do
  [ -f "$MODEL_DIR/$f" ] || missing+=("$f")
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "fetch_model: archive extracted but required files are missing: ${missing[*]}" >&2
  echo "  MODEL_DIR now holds:" >&2
  ls -la "$MODEL_DIR" >&2
  exit 1
fi

echo "fetch_model: artifact ready at $MODEL_DIR"

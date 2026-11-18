#!/usr/bin/env bash
# Software Bill of Materials for the gpuq package (WEB-1164), in CycloneDX and SPDX.
#
#   scripts/sbom.sh [out_dir]       writes <out_dir>/gpuq.cdx.json and gpuq.spdx.json (default: sbom/)
#
# The SBOM is read from an installed copy of the package, not from setup.py, so that it carries each
# package's license (from its *.dist-info metadata). The package is installed as it ships: no [dev]
# extra, into a throwaway directory. gpuq has no runtime Python dependencies today, so the SBOM lists
# gpuq itself; anything added to install_requires later shows up automatically.
#
# Uses `syft` when it is on PATH, else the pinned anchore/syft image. Prints a markdown license summary
# on stdout (CI appends it to the job summary); everything else goes to stderr. Never fails on what the
# summary finds, which is informational.

set -euo pipefail

SYFT_VERSION="${SYFT_VERSION:-v1.55.0}"
PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/sbom}" && cd "${1:-$ROOT/sbom}" && pwd)"
SOURCE_VERSION="${SOURCE_VERSION:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)}"
SITE="$(mktemp -d)"
trap 'rm -rf "$SITE"' EXIT

# File entries would add hundreds of components of no use for licensing or vulnerability matching.
export SYFT_FILE_METADATA_SELECTION=none

syft() {
  if type -P syft >/dev/null; then
    command syft "$@"
  else
    # As the caller, so the SBOMs are not left owned by root. syft needs a writable HOME and TMPDIR,
    # otherwise it writes an empty file and (with -q) says nothing.
    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -e SYFT_FILE_METADATA_SELECTION -e HOME=/work -e TMPDIR=/work --tmpfs /work:mode=1777 \
      -v "$SITE:$SITE:ro" -v "$OUT:$OUT" \
      "anchore/syft:$SYFT_VERSION" "$@"
  fi
}

echo "Installing gpuq into $SITE" >&2
"$PYTHON" -m pip install --quiet --disable-pip-version-check --target "$SITE" "$ROOT" >&2

echo "gpuq SBOM from the installed package" >&2
syft scan "dir:$SITE" -q \
  --source-name gpuq --source-version "$SOURCE_VERSION" \
  -o "cyclonedx-json=$OUT/gpuq.cdx.json" -o "spdx-json=$OUT/gpuq.spdx.json"

# License summary: per SBOM, how many packages carry each license, and which carry none.
license_summary() {
  local name="$1" file="$2"
  jq -r --arg name "$name" '
    [ .components[]? | select(.type == "library") |
      { ref: ((if .group then .group + "/" else "" end) + .name + "@" + (.version // "?")),
        licenses: [ .licenses[]? | (.license.id // .license.name // .expression) ] } ] as $pkgs
    | ($pkgs | map(select(.licenses == []))) as $unknown
    | "### \($name): \($pkgs | length) packages\n",
      "| License | Packages |", "|---|---|",
      ( [ $pkgs[] | .licenses[] ] | group_by(.) | map({ l: .[0], n: length }) | sort_by(-.n)[]
        | "| \(.l) | \(.n) |" ),
      ( if $unknown == [] then empty
        else "\n<details><summary>\($unknown | length) without a license</summary>\n",
             ($unknown[] | "- `\(.ref)`"), "\n</details>" end ),
      ""
  ' "$file"
}

license_summary gpuq "$OUT/gpuq.cdx.json"
echo "SBOMs written to $OUT" >&2

#!/usr/bin/env bash
set -euo pipefail
cd /opt/coincharge-bot

run_one () {
  local SITE="$1"
  local COLLECTION="$2"
  local SITEMAP_INDEX="$3"

  echo "=== $(date -Is) ingest site=$SITE collection=$COLLECTION ==="

  INCREMENTAL=1 PRUNE=1 \
  docker compose run --rm \
    -e SITE="$SITE" \
    -e COLLECTION="$COLLECTION" \
    -e SITEMAP_INDEX="$SITEMAP_INDEX" \
    ingest
}

# coincharge
run_one "coincharge.io" "kb_coincharge" "https://coincharge.io/sitemap_index.xml"

# coinsnap
run_one "coinsnap.io" "kb_coinsnap" "https://coinsnap.io/sitemap_index.xml"

# coinpages
run_one "coinpages.io" "kb_coinpages" "https://coinpages.io/sitemap_index.xml"

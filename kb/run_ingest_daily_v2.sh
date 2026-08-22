#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/coincharge-bot"
LOGDIR="${ROOT}/kb/reports"
mkdir -p "$LOGDIR"

# Simple lock to prevent overlapping runs
LOCKFILE="/tmp/coincharge-ingest.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[ingest] another run is still active, exiting."
  exit 0
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="${LOGDIR}/ingest_${ts}.log"

cd "$ROOT"

echo "[ingest] start ${ts}" | tee -a "$log"
echo "[ingest] docker compose project: $(basename "$ROOT")" | tee -a "$log"

run_one () {
  local site="$1"
  local sitemap="$2"
  local collection="$3"

  echo "" | tee -a "$log"
  echo "[ingest] site=${site} collection=${collection} sitemap=${sitemap}" | tee -a "$log"

  # Incremental ingest relies on persisted state under ./kb/state (mounted to /app/state)
  docker compose run --rm \
    -e SITE="$site" \
    -e SITEMAP_INDEX="$sitemap" \
    -e COLLECTION="$collection" \
    -e INCREMENTAL=1 \
    ingest 2>&1 | tee -a "$log"
}

# Coincharge
run_one "coincharge.io" "https://coincharge.io/sitemap_index.xml" "kb_coincharge_v2"

# Coinsnap
run_one "coinsnap.io" "https://coinsnap.io/sitemap_index.xml" "kb_coinsnap_v2"

# Coinpages
run_one "coinpages.io" "https://coinpages.io/sitemap_index.xml" "kb_coinpages_v2"

echo "" | tee -a "$log"
echo "[ingest] done ${ts}" | tee -a "$log"

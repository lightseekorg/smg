#!/usr/bin/env bash
# Dump the last worker log snapshot and list all log files.
#
# Usage: ci_dump_worker_logs.sh <log_dir> [artifact_name]
#
# Arguments:
#   log_dir        Directory containing worker log files
#   artifact_name  (optional) Artifact name to display in the file listing

set -euo pipefail

LOG_DIR="${1:?Usage: ci_dump_worker_logs.sh <log_dir> [artifact_name]}"
ARTIFACT="${2:-$LOG_DIR}"

if [ ! -d "$LOG_DIR" ]; then exit 0; fi

SEP="============================================================"
DASH="------------------------------------------------------------"

# Dump last worker log (most recent = last retry of the failed model)
LAST_LOG=$(ls -t "$LOG_DIR"/worker-*.log 2>/dev/null | head -1 || true)
if [ -n "$LAST_LOG" ]; then
    TOTAL=$(wc -l < "$LAST_LOG")
    SHOW=$(( TOTAL < 200 ? TOTAL : 200 ))
    echo ""
    echo "$SEP"
    echo "Last worker log: $(basename "$LAST_LOG") (last $SHOW of $TOTAL lines)"
    echo "$DASH"
    tail -200 "$LAST_LOG"
    echo "$SEP"
fi

# List all log files with sizes
echo ""
echo "$SEP"
echo "All worker logs in artifact: $ARTIFACT"
echo "$DASH"
ls -lhS "$LOG_DIR"/*.log 2>/dev/null || echo "  (none)"
echo "$SEP"

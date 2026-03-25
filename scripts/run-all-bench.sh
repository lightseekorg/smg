#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../target/mesh-bench/results.txt"
mkdir -p "$(dirname "$OUT")"

echo "# Mesh Delta Encoding Benchmark — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$OUT"
echo "" | tee -a "$OUT"
echo "3 gateways, cache_aware policy, delta encoding enabled" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Scenario | Policy Total | Batches | Avg Batch | Avg Round | Worker Total | Registered |" | tee -a "$OUT"
echo "|----------|-------------|---------|-----------|-----------|-------------|------------|" | tee -a "$OUT"

for scenario in "10 100 30" "20 100 30" "40 100 30" "20 50 30" "20 200 30" "20 500 30" "20 200 120"; do
    pkill -f mock_worker.py 2>/dev/null || true
    pkill -f 'smg.*mesh' 2>/dev/null || true
    sleep 3
    result=$(bash "$DIR/mesh-bench-one.sh" $scenario 2>/dev/null | grep '^|' || echo "| $scenario | ERROR | | | | | |")
    echo "$result" | tee -a "$OUT"
done

pkill -f mock_worker.py 2>/dev/null || true
pkill -f 'smg.*mesh' 2>/dev/null || true
echo "" | tee -a "$OUT"
echo "Done. Results at $OUT"

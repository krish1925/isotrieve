#!/bin/bash
set -u
cd /Users/kpatel/Desktop/agent-communication
VERIFY_ROOT="$(cat /tmp/verify_root)"
source "$VERIFY_ROOT/_env/lane.env"
export HF_HUB_OFFLINE=1
PY=.agent-venv/bin/python
B=benchmarks/run_benchmark.py
OUT="$VERIFY_ROOT/P6/out"; mkdir -p "$OUT"
lane(){ local name="$1"; shift; { "$@"; echo "EXIT=$?"; } > "$VERIFY_ROOT/P6/lane_$name.log" 2>&1; }
# Lane A: claim 1+2 — scifact ridge+lowrank K=4000, 3 seeds (sequential within lane; cache shared)
lane A bash -c '
for ad in ridge lowrank; do
  /Users/kpatel/Desktop/agent-communication/.agent-venv/bin/python /Users/kpatel/Desktop/agent-communication/benchmarks/run_benchmark.py --dataset scifact --adapter $ad --k 4000 --seeds 0 1 2 --out-dir '"$OUT"' --cache-dir /Users/kpatel/Desktop/agent-communication/benchmarks/.embed_cache
done'
# Lane B: claim 4 — bge→e5 same-dim K=2000, 3 seeds
lane B bash -c '
/Users/kpatel/Desktop/agent-communication/.agent-venv/bin/python /Users/kpatel/Desktop/agent-communication/benchmarks/run_benchmark.py --dataset scifact --adapter ridge --k 2000 --seeds 0 1 2 --source-model BAAI/bge-large-en-v1.5 --target-model intfloat/e5-large-v2 --out-dir '"$OUT"' --cache-dir /Users/kpatel/Desktop/agent-communication/benchmarks/.embed_cache'
# Lane C: claims 5+6 — domain matrix scifact+fiqa K=500 max-docs 2000, 2 seeds, ridge+lowrank
lane C bash -c '
for ds in scifact fiqa; do
  for ad in ridge lowrank; do
    /Users/kpatel/Desktop/agent-communication/.agent-venv/bin/python /Users/kpatel/Desktop/agent-communication/benchmarks/run_benchmark.py --dataset $ds --adapter $ad --k 500 --seeds 0 1 --max-docs 2000 --out-dir '"$OUT"' --cache-dir /Users/kpatel/Desktop/agent-communication/benchmarks/.embed_cache
  done
done'
# Lane D: claims 7+8 — probes (fast, offline) + K-sweep
lane D bash -c '
for ds in code legal; do
  /Users/kpatel/Desktop/agent-communication/.agent-venv/bin/python /Users/kpatel/Desktop/agent-communication/benchmarks/run_benchmark.py --dataset $ds --adapter lowrank --k 500 --seeds 0 1 --out-dir '"$OUT"' --cache-dir /Users/kpatel/Desktop/agent-communication/benchmarks/.embed_cache
done
/Users/kpatel/Desktop/agent-communication/.agent-venv/bin/python /Users/kpatel/Desktop/agent-communication/benchmarks/run_benchmark.py --dataset scifact --adapter ridge --k 500 1000 2000 --seeds 0 --out-dir '"$OUT"' --cache-dir /Users/kpatel/Desktop/agent-communication/benchmarks/.embed_cache'

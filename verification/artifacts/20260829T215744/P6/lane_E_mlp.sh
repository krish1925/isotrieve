#!/bin/bash
source "/Users/kpatel/Desktop/agent-communication/verification/artifacts/20260829T215744/_env/lane.env"; export HF_HUB_OFFLINE=1
cd /Users/kpatel/Desktop/agent-communication
.agent-venv/bin/python benchmarks/run_benchmark.py --dataset scifact --adapter mlp --k 4000 --seeds 0 1 2 --out-dir "/Users/kpatel/Desktop/agent-communication/verification/artifacts/20260829T215744/P6/out" --cache-dir benchmarks/.embed_cache
echo "EXIT=$?"

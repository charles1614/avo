#!/usr/bin/env bash
# Preflight a remote GPU host for AVO evals.
# Usage: scripts/setup_remote.sh <ssh-host> ["<env_activate command>"]
set -euo pipefail

HOST=${1:?usage: setup_remote.sh <ssh-host> ["<env_activate>"]}
ACTIVATE=${2:-true}

run() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$ACTIVATE && $1"; }

echo "== $HOST: connectivity =="
ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'echo ok: $(hostname)'

echo "== nvcc =="
run 'nvcc --version | tail -1' || echo "WARN: nvcc not found (needed to build kernels)"

echo "== ninja =="
run 'ninja --version' || echo "WARN: ninja not found (pip install ninja) — cpp_extension needs it"

echo "== rsync =="
run 'rsync --version | head -1' || echo "ERROR: rsync required"

echo "== python / torch / GPU =="
run 'python3 - <<EOF
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name} sm_{p.major}{p.minor}, {p.total_memory/2**30:.0f} GiB")
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except ImportError:
    print("flash_attn: not installed (FA2 baseline will be skipped)")
try:
    import flash_attn_interface; print("flash_attn_interface (FA3): present")
except ImportError:
    print("flash_attn_interface (FA3): not installed")
EOF'

echo "== GPU idle check =="
run 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | sed "s/^$/  (idle)/"' || true

echo "== scratch dir =="
run 'mkdir -p ~/avo_scratch/build_cache && echo "~/avo_scratch ready"'

echo "== all checks done =="

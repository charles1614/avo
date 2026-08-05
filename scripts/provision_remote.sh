#!/usr/bin/env bash
# Provision a clean GPU host for AVO evals (idempotent):
#   - ~/avo_scratch + build cache
#   - dedicated venv at ~/avo_scratch/venv
#   - torch wheel matching the host's CUDA toolkit, plus ninja + numpy
# Usage: scripts/provision_remote.sh <ssh-host> [torch-wheel-tag]
#   torch-wheel-tag: cu126 | cu128 | cu130 | ... (default: auto-detect from
#   the newest /usr/local/cuda-* toolkit on the host)
# Afterwards, set in your config:
#   env_activate: "export PATH=/usr/local/cuda/bin:$PATH && source ~/avo_scratch/venv/bin/activate"
set -euo pipefail

HOST=${1:?usage: provision_remote.sh <ssh-host> [torch-wheel-tag]}
TAG=${2:-auto}

run() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$1"; }

echo "== $HOST: connectivity =="
run 'echo ok: $(hostname)'

echo "== CUDA toolkit =="
CUDA_DIR=$(run 'ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true')
if [ -z "$CUDA_DIR" ]; then
    echo "ERROR: no /usr/local/cuda-* toolkit on $HOST — install the CUDA toolkit first"
    exit 1
fi
CUDA_VER=$(basename "$CUDA_DIR" | sed 's/cuda-//')
echo "found CUDA $CUDA_VER at $CUDA_DIR"

if [ "$TAG" = "auto" ]; then
    case "$CUDA_VER" in
        13.*) TAG=cu130 ;;
        12.8*|12.9*) TAG=cu128 ;;
        12.*) TAG=cu126 ;;
        *) echo "ERROR: cannot map CUDA $CUDA_VER to a torch wheel tag; pass one explicitly"; exit 1 ;;
    esac
fi
echo "torch wheel tag: $TAG"

echo "== scratch + venv =="
run 'mkdir -p ~/avo_scratch/build_cache'
run 'test -x ~/avo_scratch/venv/bin/python || python3 -m venv ~/avo_scratch/venv'
run '~/avo_scratch/venv/bin/pip install --quiet --upgrade pip'

echo "== torch ($TAG) + ninja + numpy (skipped if already satisfied) =="
run "~/avo_scratch/venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/$TAG"
run '~/avo_scratch/venv/bin/pip install --quiet ninja numpy'

echo "== verify =="
run 'export PATH=/usr/local/cuda/bin:$PATH && ~/avo_scratch/venv/bin/python - <<EOF
import shutil, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch cannot see the GPU (driver too old for this wheel?)"
p = torch.cuda.get_device_properties(0)
print(f"gpu: {p.name} sm_{p.major}{p.minor}, {p.total_memory/2**30:.0f} GiB")
print("nvcc on PATH:", bool(shutil.which("nvcc")))
EOF'

echo "== done. config snippet =="
cat <<'SNIP'
runner:
  kind: ssh
  host: <this host>
  scratch: "~/avo_scratch"
  env_activate: "export PATH=/usr/local/cuda/bin:$PATH && source ~/avo_scratch/venv/bin/activate"
SNIP
echo "(then: bash scripts/setup_remote.sh <host> \"<that env_activate>\" to preflight)"

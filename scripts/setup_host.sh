#!/usr/bin/env bash
# Set up THIS machine (a GPU host where the repo is cloned) for AVO runs:
# installs a CUDA-matched, version-pinned torch + ninja + numpy into the
# project venv and verifies the GPU is visible. Idempotent.
#
# Usage (from the repo root, after `uv sync --extra dev` or a pip install):
#   bash scripts/setup_host.sh [torch-wheel-tag]
#   torch-wheel-tag: cu126 | cu128 | cu130 (default: auto-detect from the
#   newest /usr/local/cuda-* toolkit)
#
# Reminder: launch `avo` with nvcc on PATH, e.g.
#   export PATH=/usr/local/cuda/bin:$PATH
set -euo pipefail

TAG=${1:-auto}
# Pinned: hosts set up months apart must produce comparable numbers.
TORCH_VERSION=${TORCH_VERSION:-2.13.0}

if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: no .venv here — run 'uv sync --extra dev' (or create a venv) first"
    exit 1
fi
if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install --python .venv/bin/python"
else
    INSTALL=".venv/bin/pip install --quiet"
fi

echo "== CUDA toolkit =="
CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CUDA_DIR" ]; then
    echo "ERROR: no /usr/local/cuda-* toolkit found — install the CUDA toolkit first"
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

echo "== installing torch==$TORCH_VERSION ($TAG) + ninja + numpy =="
$INSTALL torch==$TORCH_VERSION --index-url "https://download.pytorch.org/whl/$TAG"
$INSTALL ninja numpy

echo "== verify =="
PATH="$CUDA_DIR/bin:$PATH" .venv/bin/python - <<'EOF'
import shutil, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch cannot see the GPU (driver too old for this wheel?)"
p = torch.cuda.get_device_properties(0)
print(f"gpu: {p.name} sm_{p.major}{p.minor}, {p.total_memory/2**30:.0f} GiB")
print("nvcc found:", bool(shutil.which("nvcc")))
EOF

echo "== done. run avo with nvcc on PATH: export PATH=$CUDA_DIR/bin:\$PATH =="

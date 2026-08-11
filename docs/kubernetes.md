# Running AVO on a Kubernetes GPU cluster

Two topologies work. **Pod-per-route is strongly recommended**: the container
boundary gives you the filesystem isolation that bubblewrap provides on bare
metal — usually *better*, since containers typically cannot create user
namespaces at all (so `sandbox: auto` correctly degrades to `none`, and that
is fine here because the pod already is the sandbox).

## A. Pod per route (recommended, 8 pods on an 8-GPU node)

```yaml
resources:
  limits:
    nvidia.com/gpu: 1          # exclusive card per pod via the device plugin
```

Per-pod config:

```yaml
runner:
  kind: local
  python: .venv/bin/python
  sandbox: none                # the container IS the isolation boundary
  # cuda_device unset: the device plugin already exposes exactly one GPU,
  # which torch sees as device 0
  gpu_lock: ""                 # nothing else shares this card => no lock needed
```

Isolation properties you get for free: separate `/tmp`, separate filesystem,
separate PID namespace, no visibility of peer routes' `runs/` — the four R3
integrity incidents (peer workspace reads, solution copying, shared `/tmp`
leakage) are structurally impossible.

## B. One pod, all 8 GPUs, 8 routes inside it

```yaml
resources:
  limits:
    nvidia.com/gpu: 8
```

Give each route its own card and its own lock:

```yaml
# route k (k = 0..7)
runner:
  kind: local
  cuda_device: "0"             # ...through "7"
  gpu_lock: "/tmp/avo_gpu{device}.lock"   # default; per-GPU, so routes on
                                          # different cards never serialize
  sandbox: auto                # bwrap if the container allows it (rare)
```

`cuda_device` is exported as `CUDA_VISIBLE_DEVICES` to eval/profile
subprocesses, so each harness sees its assigned card as device 0. Caveat: in
this topology the routes share a filesystem, so **integrity depends on bwrap
being usable inside the container** (needs unprivileged user namespaces —
typically blocked by the default seccomp/AppArmor profile). If bwrap does not
work, prefer topology A for comparative experiments.

## When bwrap cannot work (no CAP_SYS_ADMIN, no unprivileged userns)

Most hardened containers block mount namespaces, so `bwrap` fails with
`Failed to make / slave: Permission denied` and `proot`/`firejail` are equally
unavailable. Pick in this order:

1. **One route per container/pod** — the container *is* the sandbox, no
   capabilities involved. This is the only option that isolates by
   construction; use it for any comparative experiment.
2. **`sandbox: uid`** (single container, several routes) — if the framework
   runs as root *inside* the container, each route's shell is dropped to a
   distinct unprivileged uid with `setpriv` and each run dir is `0700` owned
   by that uid, so peers are unreadable. Needs **no capabilities and no user
   namespaces** — only root inside the container, which locked-down setups
   usually still allow. Also pins a private `TMPDIR` per route.
3. **`sandbox: none`** — accept no isolation; valid for a *single* route only.

Set **`sandbox: require`** for multi-route runs: the controller then refuses
to start when neither mechanism engages, instead of degrading to a warning.
That warning-and-continue behaviour is what let a route in one experiment
bootstrap from a previous run's `/tmp` residue and produce an invalid result.

Even with no isolation, the framework now keeps its own temp files in
`runs/<id>/tmp` (0700) rather than a shared `/tmp`, and scans `/tmp` at
startup for readable kernel/eval leftovers an agent could bootstrap from.
`LD_PRELOAD` interception was considered and rejected: it only covers
glibc-linked binaries (a static binary or a raw syscall walks straight past
it), so it would provide the *appearance* of isolation without the guarantee.

## Cluster checklist

1. **Persistent storage for `runs/`** — pods are ephemeral; without a PVC a
   restart loses the lineage. Mount a PVC at the repo's `runs/` (and ideally
   `knowledge_base/external/`). Runs are resumable: `avo resume --run runs/<id>
   --confirm-spend` after a pod restart, and the interrupted step's work is
   preserved.
2. **Fetch the knowledge base once** into the PVC:
   `python scripts/fetch_kb.py --from-manifest && python scripts/fetch_nvidia_docs.py`.
   The controller's preflight warns if MANIFEST sources are missing.
3. **nvcc must be present in the image** — the harness compiles kernels at eval
   time. Use a CUDA *devel* image (`nvidia/cuda:13.x-devel-*`), not `runtime`,
   and keep `/usr/local/cuda/bin` on `PATH`.
4. **Profiling needs permissions** — `ncu` requires
   `NVreg_RestrictProfilingToAdminUsers=0` on the node (a node-level, not
   pod-level, setting) and often `securityContext.capabilities.add:
   ["SYS_ADMIN"]`. Without it the two-stage profiler still returns its
   torch.profiler stage; only the ncu deep metrics are skipped.
5. **Timing accuracy** — request whole GPUs. MIG slices and time-slicing make
   benchmark numbers non-comparable; the busy-guard will also refuse to bench
   when it sees foreign memory on the card.
6. **No node-level GPU sharing across pods** unless you intend it: the GPU lock
   is a *file* lock, so it only coordinates processes that share a filesystem.
   Two pods on the same physical card cannot see each other's lock — rely on
   the device plugin for exclusivity instead.
7. **Long runs** — set generous `activeDeadlineSeconds`/no eviction for the
   evolution pods, or lean on resume. Budget caps (`max_usd`,
   `max_wall_clock_s`) still bound each run.

## Container-specific fixes already in the framework

- The GPU busy-guard is **memory-based, not PID-based**: `nvidia-smi` reports
  host PIDs while `os.getpid()` returns the namespace PID, so the old
  PID-comparison marked the run's *own* process as foreign and failed every
  eval inside a container.
- `runner.cuda_device` pins evals to one card; `gpu_lock` templates `{device}`
  so per-card locks are the default.
- The eval-cache identity deliberately ignores `cuda_device` — identical cards
  on a node produce interchangeable results, so routes share cache entries.

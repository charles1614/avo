# AVO — agent/session guide

Reproduction of arXiv:2603.24517 (LLM agent as evolutionary variation operator
over CUDA kernels). Read README.md for user-facing docs; this file is the
working contract for coding sessions in this repo.

## Commands

```bash
uv run pytest                        # 74 offline tests — must stay green, keyless, networkless
avo eval-once --config <cfg>         # score a workspace (no LLM)
avo run|resume ... --confirm-spend   # evolution (LLM spend)
avo dashboard --watch 30 --open      # live run dashboard (no LLM)
avo export --run runs/<id>           # package a run into committable results/
python scripts/run_kernelbench.py --config configs/kernelbench_h100.yaml \
    --problems level1 --dry|--confirm-spend    # benchmark campaigns
```

## Non-negotiable invariants

1. **Cost safety is structural.** Only `avo run`, `avo resume`,
   `scripts/llm_smoke.py`, and `scripts/run_kernelbench.py` may reach an LLM,
   and all require `--confirm-spend`. Never add an LLM call to any other code
   path; never read API keys outside env vars; tests must never need a key.
2. **The framework never trusts the agent.** Scoring harnesses are staged
   pristine from `tasks/<task>/harness/` on every eval; authoritative
   references live harness-side (or inside cache-keyed params for
   KernelBench); correctness input seeds derive from the workspace content
   hash. Don't weaken any of this for convenience.
3. **One controller per run dir** — enforced by `runs/<id>/.lock`. After
   killing a run process, verify it's dead (`kill -0`) before resuming; a
   silent kill failure once put two agents in one workspace.
4. **Never cache infrastructure failures** (`stage: "harness"`); compile/
   correctness/bench failures are deterministic and cacheable
   (`avo/eval/scoring.py:cacheable`).
5. **Editing any `tasks/*/harness/` file invalidates that task's eval cache**
   (harness tree is hashed into the key) — running evolutions will re-evaluate
   once per workspace state afterwards. Fine, but expect it.
6. Path-like `runner.python` values are resolved with `absolute()`, not
   `resolve()` — a venv's python is a symlink and following it bypasses the
   venv. Don't "fix" that.
7. **Concurrent runs on one GPU are safe only because of `runner.gpu_lock`**
   (default `/tmp/avo_gpu.lock`): eval subprocesses are serialized across
   instances (timing accuracy + memory), LLM turns overlap. The lock is
   acquired before the eval's timeout clock starts. Don't remove it, and
   keep one shared path per GPU.
8. **Multi-route filesystem integrity depends on `runner.sandbox`** (bwrap):
   shell/gpu_shell run in a namespace that blanks `runs/` and re-exposes only
   the calling workspace + a private /tmp. A deny-list alone CANNOT isolate
   routes (shell is a full language) — observed: agents copying peers'
   solutions. Needs bubblewrap on the host; `auto` warns and degrades if absent.
9. **Scoring results carry a per-eval nonce** (`--result-token`, echoed in
   `meta`). Candidate code is loaded IN-PROCESS by scoring harnesses (a .so,
   or `model_new.py`) and can write `result.json` then `os._exit(0)`; a
   correct+positive result without the matching token is rejected as forged.
   Any new scoring harness MUST echo it.
10. **Harnesses re-verify correctness AFTER benchmarking** with a fresh seed —
   a memoizing candidate passes pre-bench checks then "runs" in ~0 ms on the
   repeated identical timing calls. Don't drop this recheck.
11. **Task scoring must forbid delegating to the thing being optimized.** The
   attention harness scans source for fused-attention APIs
   (`tasks/attention_cuda/harness/checks.py`) and scores 0 — else an agent
   calls SDPA/cuDNN and measures the library, not itself. New kernel tasks
   need the analogous ban. (KernelBench is exempt: better library calls are a
   legitimate optimization there.)

## Architecture in one breath

`avo/evolution/controller.py` runs the loop (step → gate → commit/fail;
budgets; resume; supervisor). `avo/agent/variation.py` is one variation step
(tool loop until `submit` or budget). `avo/agent/tools.py` defines the tool
registry. `avo/llm/` = canonical message types + Anthropic/OpenAI-compat
adapters (thinking blocks round-trip on Anthropic, are omitted on
OpenAI-compat). `avo/eval/` = Local/SSH runners + content-hash cache.
`avo/evolution/lineage.py` = git-backed versions (tags `vNNNN`, score
trailers). Tasks are directories with `task.yaml` + `seed/` + `harness/`
following the result-JSON contract (`{"correct", "score", "configs", ...}`).

## Environments & hardware

- Primary topology: framework runs ON the GPU machine, `runner.kind: local`
  (`scripts/setup_host.sh` installs pinned torch 2.13.0 matched to the local
  CUDA toolkit; launch with `/usr/local/cuda/bin` on PATH). ssh topology is
  the documented alternative (`provision_remote.sh` / `setup_remote.sh`).
- Provider: DeepSeek via OpenAI-compat (`.env` → `DEEPSEEK_API_KEY`,
  gitignored). Thinking mode via `llm.extra_body`.
- Known 3090 Ti quirks: gnome-remote-desktop holds a ~266 MiB GPU context
  (busy-guard tolerates it); some KernelBench v0.1 problems need >18 GiB and
  only fit 80 GB cards.

## Reproducibility rules

`knowledge_base/external/` is manifest-pinned (`MANIFEST.json`;
`fetch_kb.py --from-manifest` restores exact commits;
`check_kb_freshness.py` audits). Update the KB only between runs, never
mid-run. Run results are published via `avo export` into `results/` (the
nested-git workspace is exported as `workspace.bundle`). NVIDIA doc text in
`knowledge_base/external/nvidia_docs/` is committed for private-repo
reproducibility — `git rm -r --cached` it before the repo ever goes public.

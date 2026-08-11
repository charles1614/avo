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
9. **Integrity lives in `harness_lib/avo_harness`, never in task code.** The
   runners stage it into every eval's `harness/` dir, so `import avo_harness`
   works locally and on remote hosts. It owns result tokens, structured
   `fail()`/`write_result()`, the container-safe GPU busy guard, the
   banned-API scan, CUDA-event timing, and `run_scoring()`, which fixes the
   sequence *ban scan → load → correctness → benchmark → post-bench recheck →
   tokened write*. New tasks call `run_scoring` with hooks and supply task
   logic only. **Never re-implement a guard inside a task harness** — that is
   how a task silently ships without one (it happened: the busy guard was
   duplicated and drifted).
   - token: candidate code loads IN-PROCESS (a .so, `model_new.py`) and could
     write `result.json` + `os._exit(0)`; positive scores without it are forged.
   - post-bench recheck: catches candidates that memoize and time ~0 ms.
   - banned APIs: per task via `task_params.banned_apis`. KernelBench is
     deliberately exempt — better library calls are legitimate there.
10. **KernelBench models run in `.eval()` with reference weights transferred**
   — 18 problems use Dropout (random per call) and 30 use BatchNorm; train
   mode makes legitimate implementations fail correctness at random.

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

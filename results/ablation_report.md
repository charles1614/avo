# Ablation report — framework design & reasoning effort on attention-kernel evolution

RTX 3090 Ti (sm_86), BF16 attention fwd, head_dim 128, grid: S∈{1k,2k,4k,8k} × {causal, non-causal},
total tokens 8192. Model: `deepseek-v4-flash` (official API). Baselines on identical grid/timing:
**SDPA-flash 70.4**, cuDNN 63.0, efficient 47.6 geomean TFLOPS. Seed: deliberately scalar kernel, ~2.37.

## The four runs

| run | framework state | effort | best TFLOPS | ×seed | %SDPA-flash | commits / steps | cost |
|---|---|---|---:|---:|---:|---|---:|
| baseline (08-05) | fixes arrived mid-run; workspace reset on failure | high (from step 7) | 6.93 | 2.9× | 10% | 7 / 24 | $10.76 |
| arm C-old (08-08, stopped early) | + thinking, streaming, 28KB patch carry | max | 26.38 | 11.2× | 37% | 1 / 6 | ~$3.1 |
| **fresh B** | full: **persistent workspace, revert, auto-profile** | **high** | **57.69** | 24.4× | 82% | 6 / 13 | $5.01 |
| **fresh C** | full (same) | **max** | **58.35** | **24.7×** | **83%** | 2 / 12 | $5.01 |

## Finding 1 — framework design dominates model effort

Identical model, identical task: the final framework produced **8.4× the baseline's score at half the cost**.
The decisive change was **workspace persistence** (paper-faithful "internal search trajectory"): failed
tensor-core attempts stopped being rebuilt from prompt fragments and became debuggable artifacts.
Fresh B's v0001 commit message is the direct evidence — it *names the bug it fixed* in the kernel it
inherited ("O-rescale was multiplying zeros since PV ran after all 4 tiles").

## Finding 2 — champion auto-profiling steered the mid-game

Fresh B's v0002/v0003 are verbatim responses to the injected ncu diagnosis (40%/40% SOL, latency-bound →
"shrink smem so 2 blocks/SM: occupancy 8.3%→16.4%" → "3 blocks/SM, 12 warps in flight"). Score during
the profile-guided occupancy ladder: 32.8 → 45.8 → 51.3.

## Finding 3 — reasoning effort: equal endpoints, opposite risk profiles

Commit trajectories at matched $5 (spend at each commit):

| | fresh B (high) | fresh C (max) |
|---|---|---|
| commits ($ at commit) | 32.8 ($2.33) → 45.8 ($2.53) → 51.3 ($2.60) → 53.8 ($2.93) → 57.3 ($3.84) → 57.7 (~$4.1) | 45.4 ($4.74) → 58.3 ($4.90) |
| character | steady compounding from step 5 | nothing until step 10, then two giant leaps |
| risk | always something banked | a cap $0.30 earlier ⇒ recorded score 2.37 (total failure) |

Endpoints differ by 1.1% — inside the ~1% run-to-run noise floor (seed measured 2.351–2.368 across runs).
**Interpretation: high + iteration and max + contemplation reach the same place at this budget, but max's
value is concentrated in the final steps.** Prefer `high` under tight budgets; `max` only when the budget
can survive a long silent phase. Notably, C's v0002 bundled in one commit nearly everything B discovered
across four commits (register-resident Q/O/P, causal templating, occupancy-sized smem).

## Finding 4 — LLM call behavior (official API, from llm_metrics.jsonl)

| per-call | fresh B (high) | fresh C (max) |
|---|---:|---:|
| calls | 634 | 668 |
| reasoning chars p50 / p90 / max | 228 / 13,841 / 81,689 | 189 / 23,152 / 78,198 |
| reasoning tokens mean / p90 (server) | 1,582 / 4,865 | 2,081 / 8,614 |
| latency p50 / p90 (s) | 2.9 / 44 | 2.6 / 69 |
| capped turns (finish=length) / empty | 12 / 10 | 21 / 21 |
| tool-call rate | 97% | 95% |

`max` reasons ~1.7–1.8× longer at p90 and doubles the truncation/empty-turn rate (all recovered by the
escalation mechanism). Official-API reasoning has been observed up to **130k chars** in one turn
(arm C-old) — comparable to self-hosted deployments. Distribution is strongly bimodal in both modes:
median turns barely reason (tool execution), planning turns run 20–80k chars.

## Methodology caveats

- **n = 1 per arm** — LLM sampling variance is large; treat <15% differences as suggestive only.
- Fresh arms were restarted twice at near-zero-cost boundaries (metrics logging; seed-profiling),
  identically for both arms. The old runs received mid-run fixes and are not clean arms.
- The attention `profile.py` NVTX fix landed during the old arms (arm C-old saw one wrong-kernel profile).
- Benchmark noise floor ~1% (median-of-30, thermals); final champions deserve multi-round
  re-verification (mean±std appended below when available).

## Artifacts

Each run's full package (lineage, per-config scores, every agent transcript, per-call LLM metrics,
git history bundle, final kernel source) is under `results/<run-id>/`.

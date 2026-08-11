# Experiment isolation & integrity

Running several evolution routes on one machine is only a valid *comparison*
if each route solves the problem itself. This document is the authoritative
reference for how AVO enforces that, what it cannot enforce, and how to tell
the difference afterwards.

## Why this exists (two real incidents)

| | What happened | Vector |
|---|---|---|
| **R3** | A route ran `cp runs/<peer>/workspace/attention.cu ./` and submitted a peer's solution; others read peers' `lineage.jsonl` and `git show <commit>` | peer workspace readable |
| **R4** | A route ran `ls /tmp` on its 3rd command, found a previous run's kernel sources, and reached 181 TFLOPS in 3 steps instead of deriving anything | shared `/tmp` residue |

Both produced numbers that *looked* valid. A command deny-list cannot prevent
this — the shell is a full language (`cat $(echo …)`, base64, python). Only a
filesystem boundary can.

## Mechanisms

`runner.sandbox` selects the mechanism; `auto` prefers the strongest available.

| mode | how it isolates | requires | use when |
|---|---|---|---|
| `bwrap` | mount namespace: blanks the whole `runs/` tree, re-exposes only this route's workspace, private `/tmp` | user namespaces or `CAP_SYS_ADMIN` | bare metal / permissive hosts |
| `uid` | per-route unprivileged uid (`setpriv`) + run dirs `chmod 0700` owned by that uid + private `TMPDIR` | **root inside the container; no capabilities, no userns** | locked-down containers where bwrap fails |
| `none` | nothing | — | single-route runs only |
| `require` | resolves to `bwrap` → `uid`, else **aborts the run** | — | **every multi-route experiment** |

Always the strongest thing available: **one route per container/pod**. Then the
container *is* the boundary, `sandbox: none` is honest and correct, and no
capabilities are involved.

### Why `require` matters more than the mechanism

The R4 experiment was invalidated not because isolation was missing, but
because the framework *warned and continued*. `sandbox: require` makes the
controller abort before step 1 with actionable options, so a contaminated
comparison is never produced silently.

## Choosing a configuration

Run this **inside** the container/pod that will host the routes:

```bash
python scripts/check_isolation.py
```

It prints the mechanism `auto` would pick, plus any readable `/tmp` residue.

- **reports `bwrap` or `uid`** → put `sandbox: require` in every route's config.
- **reports `none`** and you can run as root → do so; `uid` engages
  automatically.
- **reports `none`** and you cannot → prevention is impossible in one
  filesystem namespace. Run anyway if you accept the risk, and *audit every
  route* (below) before comparing. Prefer sequential routes with `/tmp`
  cleaned between them.

## Detection: auditing what isolation could not prevent

Every run ends with a verdict in `summary.json` and `logs/integrity.json`;
you can also re-check any run at any time:

```bash
avo audit --run runs/<id> --isolation none     # exit 2 if contaminated
```

Findings are classified, and only the first two condemn a run:

| kind | meaning | verdict |
|---|---|---|
| `peer_route` | read/copied another route's workspace, lineage or git history | **contaminated** |
| `foreign_tmp` | read a `/tmp` file this route never wrote | **contaminated** |
| `recon` | `ls /tmp`, `find /` — enables discovery, isn't proof of use | reported only |

Precision is deliberate: a first implementation flagged 55 "violations" in a
real run that were all the agent reading *its own* scratch files. Self-created
paths are now tracked, and all historical runs audit clean — which is what
makes a positive finding meaningful.

## Hygiene the framework does regardless of mode

- Each run writes temp files to `runs/<id>/tmp` (0700), never a shared `/tmp`,
  so AVO stops seeding the residue that caused R4.
- Run directories are `chmod 0700` (and `chown`ed to the route uid when root).
- The preflight lists readable kernel/eval residue found in `/tmp`.
- Scoring integrity is independent of all of this: the harness-level
  banned-API scan, result tokens, and the post-benchmark recheck work in every
  mode (see `harness_lib/avo_harness`).

## Rejected approaches

- **`LD_PRELOAD` syscall interception** — only covers glibc-linked dynamic
  binaries; a static binary, a Go tool, or a raw `syscall()` walks past it. It
  would give the *appearance* of isolation without the guarantee, which is
  worse than a documented `none`.
- **`proot`** (ptrace) — usually unavailable in these containers, and its
  2–5× overhead on compile/eval distorts the very timings being measured.
- **Command deny-lists as isolation** — unfixable by construction (see above);
  retained only as a speed bump against destructive commands.

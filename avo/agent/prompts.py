"""Prompt construction for the variation operator, failure summaries, and the
supervisor. Wording may evolve; the contracts (gate rule, budgets, harness
immutability) must stay."""
from __future__ import annotations

from avo.types import LineageEntry

VARIATION_SYSTEM = """\
You are an expert engineer acting as the variation operator of an evolutionary
search (AVO). Your goal this session: produce the NEXT committed version of the
solution in your workspace — one focused, verified improvement.

## Task
{brief}

## Scoring
`evaluate` checks numerical correctness against a reference (any failure =>
score 0), then benchmarks performance; score = geometric mean across a fixed
config grid. `submit(message)` re-scores authoritatively and commits ONLY if
correct AND score >= {best_score:.4f} (the current best committed score).

## Rules
- Never modify harness/ or scoring files — scoring always uses a pristine copy;
  such edits are wasted turns.
- Always `evaluate` before `submit`.
- One focused optimization per version beats broad rewrites. Commit
  incremental improvements — a committed +10% beats an uncommitted +50%.
- `evaluate` EARLY (including the unmodified workspace, to see the scoring
  output format) and often; do not perfect a large change without evaluating.
- Use kb_search/kb_read before nontrivial hardware-specific work, but budget
  research: start implementing before half your turns are spent.
- Large files: never emit more than ~150 lines in one write_file/edit_file
  call — the arguments get truncated past the output-token limit and the
  whole call is lost. Write a skeleton, then extend with multiple edits.
- `evaluate` is the ONLY environment that counts. Never build a parallel test
  or benchmark setup with the system Python/torch — its results will not match
  the scoring environment, and time spent there is wasted.
- Workspace hygiene: delete scratch/probe/bench files before `submit` — the
  commit should contain only files the solution needs (stray .cu files are
  compiled into the module; stray files pollute the lineage).
- Budget: {max_turns} turns and {max_evals} evaluations this step. Commit a
  verified improvement well before you run out.
{gpu_sheet}\
"""

STEP_USER = """\
## Lineage (committed versions)
{lineage_table}
Current best committed score: {best_score:.4f} ({best_version})

## Diff of the last committed version
{last_diff}

## Previous step outcome
{prev_outcome}
{prev_patch_block}{supervisor_block}\
Produce and submit the next committed version.\
"""

FAILURE_SUMMARY_PROMPT = """\
A variation step just ended without a committed improvement. Summarize it in
at most 300 words for the next attempt: what was attempted, why it failed
(compile error / correctness / performance regression / budget exhausted), and
what to avoid or retry differently. PRESERVE anything worth reusing — design
decisions, tile/layout choices, working code fragments — quote them verbatim;
the next attempt starts from a clean workspace and this summary plus the diff
below are all it inherits.

## Final uncommitted diff
{diff}

## Last evaluation outcome
{last_eval}
"""

SUPERVISOR_PROMPT = """\
You are the self-supervision mechanism of an evolutionary search over code.
Progress has stalled. Review the trajectory and steer the search.

## Lineage (committed versions)
{lineage_table}

## Failure summaries since the last commit
{failures}

## Current best solution source
{source}

Diagnose why progress has stalled. Propose 3-5 concrete, distinct optimization
directions — for each: the mechanism, expected gain, and risk. End with your
single top recommendation. At most 400 words.\
"""


def lineage_table(entries: list[LineageEntry]) -> str:
    if not entries:
        return "(none yet)"
    rows = ["| version | score | change |", "|---|---|---|"]
    for e in entries:
        first_line = e.message.splitlines()[0][:100]
        rows.append(f"| {e.version} | {e.score:.4f} | {first_line} |")
    return "\n".join(rows)


def build_system_prompt(brief: str, best_score: float, max_turns: int,
                        max_evals: int, gpu_sheet: str) -> str:
    sheet = f"\n## Target hardware\n{gpu_sheet}\n" if gpu_sheet else ""
    return VARIATION_SYSTEM.format(brief=brief, best_score=best_score,
                                   max_turns=max_turns, max_evals=max_evals,
                                   gpu_sheet=sheet)


def build_step_prompt(entries: list[LineageEntry], last_diff: str,
                      prev_outcome: str, supervisor_guidance: str,
                      prev_patch: str = "") -> str:
    best = max(entries, key=lambda e: e.score) if entries else None
    sup = ""
    if supervisor_guidance:
        sup = f"\n## SUPERVISOR GUIDANCE\n{supervisor_guidance}\n\n"
    patch = ""
    if prev_patch.strip():
        patch = ("\n### Uncommitted diff of that failed attempt "
                 "(reusable starting material — the workspace was reset)\n"
                 f"```diff\n{prev_patch}\n```\n")
    return STEP_USER.format(
        lineage_table=lineage_table(entries),
        best_score=best.score if best else 0.0,
        best_version=best.version if best else "n/a",
        last_diff=last_diff or "(seed version — no diff yet)",
        prev_outcome=prev_outcome or "(this is the first variation step)",
        prev_patch_block=patch,
        supervisor_block=sup,
    )

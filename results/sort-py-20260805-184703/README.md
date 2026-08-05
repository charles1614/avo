# AVO run export: sort-py-20260805-184703

- `lineage.jsonl` / `evals/` — committed versions and their full score records
- `workspace.bundle` — complete git history of the evolution (`git clone workspace.bundle kernel-history` to inspect every version)
- `final_solution/` — the best committed solution's source tree
- `baselines/`, `dashboard.html` — reference numbers and visualization

Re-verify the headline score on matching hardware:
```
avo eval-once --config <the config in config.yaml> --workspace final_solution --fresh
```
Note: the agentic evolution itself is not bit-reproducible (LLM sampling); what is reproducible is verification of every committed artifact and re-running the method.

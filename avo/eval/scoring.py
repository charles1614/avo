"""Scoring protocol and the commit gate.

Harness contract (every task follows it):
    python harness/<score_entry> --workspace <dir> --params-b64 <b64 json> --out result.json
result.json schema: see avo.types.ScoreResult. `score` must be 0.0 when
`correct` is false. The harness is always staged fresh from the pristine task
directory — the agent cannot influence scoring by editing harness files.
"""
from __future__ import annotations

import base64
import json

from avo.types import ScoreResult


def encode_params(params: dict) -> str:
    return base64.b64encode(json.dumps(params, sort_keys=True).encode()).decode()


def decode_params(b64: str) -> dict:
    return json.loads(base64.b64decode(b64))


def gate(candidate: ScoreResult, best_score: float) -> tuple[bool, str]:
    """The paper's commit rule: correct AND matches-or-improves the best
    committed score. Strict >= with no epsilon (median timing makes ties
    meaningful; an epsilon would allow slow downward drift)."""
    if not candidate.correct:
        err = candidate.error or {}
        return False, (f"REJECTED: failed correctness (stage={err.get('stage', '?')}: "
                       f"{str(err.get('detail', ''))[:300]})")
    if candidate.score >= best_score:
        return True, (f"ACCEPTED: score {candidate.score:.4f} >= "
                      f"best committed {best_score:.4f}")
    return False, (f"REJECTED: score {candidate.score:.4f} < "
                   f"best committed {best_score:.4f}")

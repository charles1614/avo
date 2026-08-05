from avo.eval.scoring import gate
from avo.types import ScoreResult, geomean


def ok(score: float) -> ScoreResult:
    return ScoreResult(correct=True, score=score)


def test_incorrect_always_rejected():
    bad = ScoreResult.failure("correctness", "wrong values")
    passed, verdict = gate(bad, best_score=0.0)
    assert not passed and "correctness" in verdict


def test_improvement_accepted():
    passed, _ = gate(ok(10.5), best_score=10.0)
    assert passed


def test_tie_accepted():
    # the paper: "matches or improves"
    passed, _ = gate(ok(10.0), best_score=10.0)
    assert passed


def test_regression_rejected():
    passed, verdict = gate(ok(9.99), best_score=10.0)
    assert not passed and "REJECTED" in verdict


def test_incorrect_high_score_still_rejected():
    cheat = ScoreResult(correct=False, score=999.0,
                        error={"stage": "correctness", "detail": "x"})
    passed, _ = gate(cheat, best_score=1.0)
    assert not passed


def test_geomean():
    assert abs(geomean([4.0, 16.0]) - 8.0) < 1e-9
    assert geomean([]) == 0.0
    assert geomean([1.0, 0.0]) == 0.0

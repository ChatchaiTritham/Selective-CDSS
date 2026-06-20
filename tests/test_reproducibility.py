"""Reproducibility unit tests for Selective-CDSS.

These tests lock the verified, headline results of the risk-controlled
selective-prediction evaluation (``run_eval.py``, deterministic at seed 42).
They serve two purposes:

1. Regression-lock the committed ``results/real_results.json`` so that any
   accidental change to the numbers reported in the manuscript is caught.
2. Optionally re-run the full pipeline (``run_eval.py``) and assert the
   regenerated metrics match the committed JSON to a numerical tolerance,
   proving the run is reproducible. That live re-run is marked ``slow`` and is
   deselected by default (it trains a 300-tree RandomForest three times over).

Run the fast lock tests:        pytest tests/test_reproducibility.py
Include the live determinism:   pytest -m slow tests/test_reproducibility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Paths / fixtures
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO_ROOT / "results" / "real_results.json"

# Numerical tolerances.
#   ABS_TOL  : float comparison for metrics stored as fractions (0..1).
#   PCT_TOL  : looser band when a value is described in the manuscript as "~X%".
ABS_TOL = 1e-9            # committed-JSON self-consistency (exact reload)
LIVE_TOL = 5e-3           # live re-run vs committed (RF/sklearn version drift)


@pytest.fixture(scope="module")
def results() -> dict:
    """The committed, verified results that back the manuscript numbers."""
    assert RESULTS_JSON.is_file(), f"missing committed results: {RESULTS_JSON}"
    return json.loads(RESULTS_JSON.read_text())


# --------------------------------------------------------------------------- #
# 0. Well-formedness / determinism config
# --------------------------------------------------------------------------- #
def test_results_json_is_wellformed(results):
    """Committed JSON parses and carries the seed-42 deterministic config."""
    cohort = results["cohort"]
    assert cohort["seed"] == 42
    assert cohort["margin"] == pytest.approx(0.10, abs=ABS_TOL)
    assert cohort["target_fnr"] == pytest.approx(0.05, abs=ABS_TOL)
    # cohort shape that the manuscript reports
    assert cohort["n"] == 1140
    assert cohort["n_per_disease"] == 380
    # split is train/cal/test = 570/285/285
    assert results["split"] == {"train": 570, "cal": 285, "test": 285}


# --------------------------------------------------------------------------- #
# 1. Clean reference: base FNR ~3.1% -> retained FNR ~1.1% at ~1.8% abstention
# --------------------------------------------------------------------------- #
def test_clean_base_fnr_about_3_1pct(results):
    clean = results["clean"]
    # 0.030927835... == 3/97 -> "~3.1%"
    assert clean["base_fnr"] == pytest.approx(0.030927835051546393, abs=ABS_TOL)
    assert clean["base_fnr"] == pytest.approx(0.031, abs=2e-3)


def test_clean_retained_fnr_about_1_1pct(results):
    clean = results["clean"]
    assert clean["retained_fnr_test"] == pytest.approx(0.010638297872340425, abs=ABS_TOL)
    assert clean["retained_fnr_test"] == pytest.approx(0.011, abs=2e-3)


def test_clean_abstention_about_1_8pct(results):
    clean = results["clean"]
    assert clean["abstention"] == pytest.approx(0.01754385964912286, abs=ABS_TOL)
    assert clean["abstention"] == pytest.approx(0.018, abs=2e-3)


def test_clean_is_risk_controlled(results):
    assert results["clean"]["risk_controlled"] is True


def test_clean_abstention_reduces_fnr(results):
    """Selective prediction must improve safety vs full automation on clean data."""
    clean = results["clean"]
    assert clean["retained_fnr_test"] < clean["base_fnr"]


# --------------------------------------------------------------------------- #
# 2. Under degradation (MCAR/MAR/MNAR @ rate 0.35, sigma 0.6)
#    - retained FNR stays <= ~2.2%
#    - abstention range ~30.9% - 37.5%
# --------------------------------------------------------------------------- #
RETAINED_FNR_CAP = 0.022 + 1e-6          # "<= ~2.2%"
ABSTENTION_LO, ABSTENTION_HI = 0.309, 0.375 + 1e-3

EXPECTED_MECH_RETAINED_FNR = {
    "mcar": 0.02197802197802198,
    "mar": 0.0,
    "mnar": 0.011363636363636364,
}
EXPECTED_MECH_ABSTENTION = {
    "mcar": 0.3508771929824561,
    "mar": 0.343859649122807,
    "mnar": 0.375438596491228,
}


@pytest.mark.parametrize("mech", ["mcar", "mar", "mnar"])
def test_degraded_retained_fnr_within_cap(results, mech):
    m = results["mechanisms"][mech]
    assert m["retained_fnr_test"] == pytest.approx(
        EXPECTED_MECH_RETAINED_FNR[mech], abs=ABS_TOL
    )
    assert m["retained_fnr_test"] <= RETAINED_FNR_CAP


@pytest.mark.parametrize("mech", ["mcar", "mar", "mnar"])
def test_degraded_abstention_in_range(results, mech):
    m = results["mechanisms"][mech]
    assert m["abstention"] == pytest.approx(
        EXPECTED_MECH_ABSTENTION[mech], abs=ABS_TOL
    )
    assert ABSTENTION_LO <= m["abstention"] <= ABSTENTION_HI


@pytest.mark.parametrize("mech", ["mcar", "mar", "mnar"])
def test_degraded_is_risk_controlled(results, mech):
    assert results["mechanisms"][mech]["risk_controlled"] is True


def test_missed_positives_drop_10_to_4(results):
    """Moderate MCAR harm analysis: full automation misses 10, selective misses 4."""
    harm = results["harm_moderate_mcar"]
    assert harm["missed_pos_full"] == 10
    assert harm["missed_pos_retained"] == 4
    assert harm["missed_pos_retained"] < harm["missed_pos_full"]


# --------------------------------------------------------------------------- #
# 3. Baseline comparison @ moderate MCAR (full / SR / split-conformal / LTT)
#    retained-FNR 10.3 / 5.2 / 4.8 / 2.2 ; coverage 100 / 65 / 74 / 65
# --------------------------------------------------------------------------- #
EXPECTED_BASELINES = {
    "Full automation": {
        "retained_fnr": 0.10309278350515463,
        "coverage": 1.0,
        "targets_fnr": False,
    },
    "Softmax-response (matched coverage)": {
        "retained_fnr": 0.05194805194805195,
        "coverage": 0.6491228070175439,
        "targets_fnr": False,
    },
    "Split-conformal (alpha=0.05)": {
        "retained_fnr": 0.04819277108433735,
        "coverage": 0.7403508771929824,
        "targets_fnr": False,
    },
    "LTT risk control (ours)": {
        "retained_fnr": 0.02197802197802198,
        "coverage": 0.6491228070175439,
        "targets_fnr": True,
    },
}


def test_baselines_target_fnr(results):
    assert results["baselines_moderate_mcar"]["target_fnr"] == pytest.approx(
        0.05, abs=ABS_TOL
    )


@pytest.mark.parametrize("method", list(EXPECTED_BASELINES))
def test_baseline_method_metrics(results, method):
    methods = {m["method"]: m for m in results["baselines_moderate_mcar"]["methods"]}
    assert method in methods, f"missing baseline method: {method}"
    got = methods[method]
    exp = EXPECTED_BASELINES[method]
    assert got["retained_fnr"] == pytest.approx(exp["retained_fnr"], abs=ABS_TOL)
    assert got["coverage"] == pytest.approx(exp["coverage"], abs=ABS_TOL)
    assert got["targets_fnr"] is exp["targets_fnr"]


def test_only_ltt_is_certified(results):
    """LTT is the only method that targets FNR with a certificate."""
    methods = {m["method"]: m for m in results["baselines_moderate_mcar"]["methods"]}
    ltt = methods["LTT risk control (ours)"]
    assert ltt.get("certified") is True
    # LTT achieves the lowest retained FNR among all baselines
    fnrs = [m["retained_fnr"] for m in results["baselines_moderate_mcar"]["methods"]]
    assert ltt["retained_fnr"] == min(fnrs)


# --------------------------------------------------------------------------- #
# 4. LogReg baseline: FNR ~38.1% and NOT certified (risk_controlled == False)
# --------------------------------------------------------------------------- #
def test_logreg_base_fnr_about_38pct(results):
    lr = results["logreg_moderate_mcar"]
    assert lr["base_fnr"] == pytest.approx(0.38144329896907214, abs=ABS_TOL)
    assert lr["base_fnr"] == pytest.approx(0.381, abs=2e-3)


def test_logreg_not_risk_controlled(results):
    """A weak base model cannot be certified -> risk_controlled must be False."""
    assert results["logreg_moderate_mcar"]["risk_controlled"] is False


# --------------------------------------------------------------------------- #
# 5. Determinism: live re-run of run_eval.py must match committed JSON.
#    Marked slow (trains RF x3 + cohort sim). Deselected by default.
# --------------------------------------------------------------------------- #
def _flatten_numeric(obj, prefix=""):
    """Yield (path, float) for every numeric leaf, skipping bools."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_numeric(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten_numeric(v, f"{prefix}[{i}]")


@pytest.mark.slow
def test_live_rerun_matches_committed(results, tmp_path, monkeypatch):
    """Re-execute the evaluation and assert regenerated metrics match committed.

    This is the true reproducibility check. It is slow because it rebuilds the
    synthetic cohort and trains the RandomForest pipeline from scratch. It is
    skipped automatically if the scientific dependencies are unavailable.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    src = REPO_ROOT / "src"
    pytest.importorskip(
        "basics_cdss",
    ) if str(src) in __import__("sys").path else None

    import sys
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import run_eval  # noqa: WPS433  (import inside test by design)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import run_eval / basics_cdss: {exc!r}")

    # Run into an isolated dir so we do not overwrite the committed artifact.
    monkeypatch.setattr(run_eval, "OUT", tmp_path, raising=True)
    run_eval.main()

    regenerated = json.loads((tmp_path / "real_results.json").read_text())

    committed_flat = dict(_flatten_numeric(results))
    regen_flat = dict(_flatten_numeric(regenerated))

    # Every committed numeric leaf must reappear within tolerance.
    mismatches = []
    for path, exp in committed_flat.items():
        if path not in regen_flat:
            mismatches.append(f"{path}: missing in regenerated")
            continue
        if abs(regen_flat[path] - exp) > LIVE_TOL:
            mismatches.append(f"{path}: {regen_flat[path]} != {exp}")
    assert not mismatches, "non-deterministic re-run:\n" + "\n".join(mismatches[:20])

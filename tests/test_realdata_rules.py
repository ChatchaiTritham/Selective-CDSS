"""The real-data rules must equal the run_ictx_r4.py rules they were lifted from."""
import ast
import math
import sys
import types
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "realdata"))
import rules as RL  # noqa: E402

NAMES = {"tau_emp", "tau_cert", "tau_selcert", "_sgr_ok", "sgr_binary"}


def _r4():
    src = (ROOT / "scripts/ictx/scripts/run_ictx_r4.py").read_text(encoding="utf-8")
    mod = ast.Module([n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name in NAMES], [])
    R = types.SimpleNamespace(fnr_at=RL.fnr_at)
    g = dict(np=np, binom=binom, math=math, R=R, ALPHA=0.05, DELTA=0.10, M0=0.10,
             GRID=np.linspace(0.01, 0.99, 99), CUTS=np.linspace(0.5, 0.99, 50))
    exec(compile(mod, "r4", "exec"), g)
    return g


def test_rules_match_r4():
    g = _r4()
    rng = np.random.default_rng(0)
    for n in (60, 300, 2000):
        y = (rng.random(n) < 0.2).astype(int)
        p = np.clip(rng.beta(2, 5, n) + 0.35 * y, 0, 1)
        assert RL.tau_emp(p, y, 0.05) == g["tau_emp"](p, y)
        assert RL.tau_cert(p, y, 0.05, 0.10) == g["tau_cert"](p, y)
        assert RL.tau_selcert(p, y, 0.05, 0.10, 0.10) == g["tau_selcert"](p, y)
        assert RL.sgr_binary(p, y, 0.05, 0.10) == g["sgr_binary"](p, y)


def test_refusal_floor():
    assert RL.refusal_floor(0.05, 0.10) == 45
    assert RL.refusal_floor(0.10, 0.10) == 22


def test_selcert_refuses_below_floor():
    # 44 retained positives with zero misses cannot certify alpha=0.05 at delta=0.10
    y = np.r_[np.ones(44, int), np.zeros(100, int)]
    p = np.r_[np.full(44, 0.99), np.full(100, 0.5)]
    assert RL.tau_selcert(p, y, 0.05, 0.10, 0.10) is None
    y2 = np.r_[np.ones(45, int), np.zeros(100, int)]
    p2 = np.r_[np.full(45, 0.99), np.full(100, 0.5)]
    assert RL.tau_selcert(p2, y2, 0.05, 0.10, 0.10) is not None

"""Regression tests for the paper's factor-aware validation contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROOT
from estimation.ml import LOAD_THRESH
from estimation.twin import CALIB, physics_cross_estimate
from scripts.benchmark import (
    _fixture_angle_valid,
    _localization,
    _protocol_splits,
    _protocol_test_mask,
    _protocols,
    _score_pred,
)


def _lab() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "analysis" / "features.csv")
    return df[df.labeled == 1].reset_index(drop=True)


def test_angle_protocol_is_eight_single_sector_folds():
    lab = _lab()
    spec = _protocols(lab)["angle_sector"]
    groups = np.asarray(spec["groups"])
    folds = _protocol_splits(len(lab), spec)
    assert len(folds) == 8
    assert {int(groups[te[0]]) for _, te in folds} == set(range(8))
    for tr, te in folds:
        assert len(np.unique(groups[te])) == 1
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_magnitude_interpolation_scores_only_bracketed_levels():
    lab = _lab()
    spec = _protocols(lab)["magnitude_interpolating"]
    groups = np.asarray(spec["groups"])
    folds = _protocol_splits(len(lab), spec)
    levels = sorted(np.unique(groups))
    assert len(folds) == len(levels) - 2
    tested = []
    for tr, te in folds:
        held = float(np.unique(groups[te])[0])
        trained = np.unique(groups[tr])
        assert float(trained.min()) < held < float(trained.max())
        assert len(np.unique(groups[te])) == 1
        tested.append(held)
    assert min(levels) not in tested
    assert max(levels) not in tested


def test_condition_protocol_remains_five_fold_without_leakage():
    lab = _lab()
    spec = _protocols(lab)["condition"]
    groups = np.asarray(spec["groups"])
    folds = _protocol_splits(len(lab), spec)
    assert len(folds) == 5
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_every_configured_fold_has_scored_test_observations():
    lab = _lab()
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    expected = {
        "random_acquisition": 5,
        "condition": 5,
        "magnitude_interpolating": 10,
        "magnitude_extrapolating": 1,
        "configuration": 4,
        "angle_sector": 8,
    }
    for name, spec in _protocols(lab).items():
        folds = _protocol_splits(len(lab), spec)
        assert len(folds) == expected[name]
        assert all(scored[te].any() for _, te in folds)


def test_six_is_loaded_but_rounding_does_not_promote_5969():
    at_boundary = pd.DataFrame({
        "cid": ["C1", "C1"], "true_mag": [LOAD_THRESH, 0.0],
        "est_mag": [LOAD_THRESH, 0.0],
    })
    assert _localization(at_boundary) == (1.0, 1)

    below = at_boundary.copy()
    below.loc[0, "est_mag"] = 5.969
    assert round(float(below.loc[0, "est_mag"]), 1) == LOAD_THRESH
    assert _localization(below) == (0.0, 1)


def test_high_total_holdout_is_exactly_ge_36():
    lab = _lab()
    spec = _protocols(lab)["magnitude_extrapolating"]
    test = _protocol_test_mask(len(lab), spec)
    total = (lab.U1_mag + lab.U2_mag).to_numpy(float)
    assert np.array_equal(test, total >= 36.0)


def test_fixture_grid_accepts_only_reachable_angles():
    assert _fixture_angle_valid(157.5)
    assert _fixture_angle_valid(67.5)
    assert not _fixture_angle_valid(157.0)


def test_every_logged_reference_angle_is_on_the_fixture_grid():
    """No condition may carry an angle the hole circle cannot produce.

    BLND04 was logged at 157.0 degrees, which is unreachable; the author
    confirmed 157.5 and the table was corrected. This guards the whole table
    against the same class of transcription error.
    """
    rows = pd.read_csv(ROOT / "datasets" / "conditions.csv")
    bad = []
    for col in ("disk1_angle_deg", "disk2_angle_deg"):
        for cid, ang in zip(rows.condition_id, rows[col]):
            if pd.notna(ang) and not _fixture_angle_valid(float(ang)):
                bad.append((cid, col, float(ang)))
    assert not bad, f"off-grid reference angles: {bad}"


def test_icm_is_scored_on_each_protocol_subset():
    lab = _lab()
    df = pd.read_csv(ROOT / "analysis" / "features.csv")
    physics = physics_cross_estimate(df)
    components = []
    for k in (1, 2):
        z = (physics[f"est_U{k}_mag"].to_numpy(float)
             * np.exp(1j * np.radians(physics[f"est_U{k}_ang"].to_numpy(float))))
        components.extend([z.real, z.imag])
    pred = np.column_stack(components)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    specs = _protocols(lab)

    interp = _score_pred(
        lab, pred,
        scored & _protocol_test_mask(len(lab), specs["magnitude_interpolating"]),
    )
    high = _score_pred(
        lab, pred,
        scored & _protocol_test_mask(len(lab), specs["magnitude_extrapolating"]),
    )
    assert interp["r2_pred"] == 0.5022
    assert high["r2_pred"] == 0.4084
    assert interp["n_points"] != high["n_points"]

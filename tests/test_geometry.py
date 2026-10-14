import numpy as np

from dbl.geometry import angdiff, bearing, circmean
from dbl.pose.extract import _heading, features_from_block, J


def test_angdiff_wraps():
    assert angdiff(170, -170) == -20
    assert angdiff(-170, 170) == 20


def test_circmean_across_wrap():
    assert abs(abs(circmean([179, -179])) - 180) < 1e-6


def test_bearing():
    assert bearing(0, 0, 1, 0) == 0
    assert bearing(0, 0, 0, 1) == 90


def test_heading_convention_facing_plus_x():
    # facing +x: left shoulder at +y, right shoulder at -y
    deg, w = _heading(np.array([[0.0, 0.2, 1.5]]), np.array([[0.0, -0.2, 1.5]]))
    assert abs(deg[0]) < 1e-6 and abs(w[0] - 0.4) < 1e-9


def _upright_player(face_deg: float):
    """Synthetic skeleton facing `face_deg`, standing (straight legs)."""
    t = np.radians(face_deg)
    fwd, left = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    pts = {
        "nose": (0.10 * fwd, 1.70), "lEar": (0.08 * left, 1.68), "rEar": (-0.08 * left, 1.68),
        "lEye": (0.05 * fwd + 0.03 * left, 1.72), "rEye": (0.05 * fwd - 0.03 * left, 1.72),
        "lShoulder": (0.2 * left, 1.45), "rShoulder": (-0.2 * left, 1.45), "neck": (0 * left, 1.50),
        "lHip": (0.12 * left, 0.95), "rHip": (-0.12 * left, 0.95), "midHip": (0 * left, 0.95),
        "lKnee": (0.12 * left, 0.50), "rKnee": (-0.12 * left, 0.50),
        "lAnkle": (0.12 * left, 0.08), "rAnkle": (-0.12 * left, 0.08),
    }
    xyz = np.array([[*pts[j][0], pts[j][1]] for j in J])[None]
    return xyz, np.full((1, len(J)), 8.0)


def test_features_recover_heading_and_straight_knees():
    for face in (0, 45, 135, -100):
        f = features_from_block(*_upright_player(face))
        assert abs(angdiff(f["torso_deg"][0], face)) < 1
        assert abs(angdiff(f["head_deg"][0], face)) < 1
        assert f["knee_deg"][0] > 175


def test_noisy_joints_are_gated():
    xyz, err = _upright_player(0)
    err[:] = 60.0  # far above the p90 error gate
    f = features_from_block(xyz, err)
    assert np.isnan(f["torso_deg"][0]) and np.isnan(f["head_deg"][0])

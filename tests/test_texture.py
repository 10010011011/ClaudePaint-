import math
import numpy as np
import pytest
from scipy import ndimage as ndi
from nsp.flatten import rgb_to_lab, lab_to_rgb
from nsp.texture import texture, load_director, DEFAULTS

def flat_img(H=160, W=160, lab=(60, 5, 5)):
    return lab_to_rgb(np.broadcast_to(np.array(lab, float), (H, W, 3)).copy())

def stroke_angle(out, base, mask=None):
    """Dominant stroke direction (math degrees, 0=right, 90=up, mod 180) + coherence of the residual."""
    R = rgb_to_lab(out)[..., 0] - rgb_to_lab(base)[..., 0]
    gx, gy = ndi.gaussian_filter(R, 1.0, order=(0, 1)), ndi.gaussian_filter(R, 1.0, order=(1, 0))
    if mask is not None:
        gx, gy = gx * mask, gy * mask
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    th_grad = 0.5 * math.atan2(2 * jxy, jxx - jyy)
    deg = (-math.degrees(th_grad + math.pi / 2)) % 180
    return deg, math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12)

def angdiff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)

def test_fixed_direction_followed():
    base = flat_img(); gl = np.ones(base.shape[:2], np.int32)
    p = load_director({"defaults": {"size": 8, "jitter": 3.0, "direction": 30, "alpha": 0.9}}, [1])
    out, info = texture(base, base, gl, p, seed=1)
    deg, coh = stroke_angle(out, base)
    assert info["strokes"] > 100
    assert angdiff(deg, 30) < 10 and coh > 0.3, (deg, coh)

def test_flow_follows_source_structure():
    H = W = 160
    ys, xs = np.mgrid[0:H, 0:W]
    # stripes constant along the up-right diagonal (math 45 deg); gradient points (1,1)/sqrt2
    L = 60 + 6 * np.sin(0.35 * (xs * 0.7071 + ys * 0.7071))
    src = lab_to_rgb(np.stack([L, np.full_like(L, 5), np.full_like(L, 5)], -1))
    base = flat_img(H, W); gl = np.ones((H, W), np.int32)
    p = load_director({"defaults": {"size": 8, "jitter": 3.0, "direction": "flow", "alpha": 0.9}}, [1])
    out, _ = texture(base, src, gl, p, seed=2)
    deg, coh = stroke_angle(out, base)
    assert angdiff(deg, 45) < 15 and coh > 0.25, (deg, coh)

def test_skip_untouched_no_bleed_mean_preserved():
    H, W = 120, 160
    a, b = (50, 20, 0), (62, -15, 10)          # dE ~ 25 apart
    lab = np.zeros((H, W, 3)); lab[:, :80] = a; lab[:, 80:] = b
    base = lab_to_rgb(lab); gl = np.ones((H, W), np.int32); gl[:, 80:] = 2
    p = load_director({"groups": {"1": {"skip": True}, "2": {"size": 8, "jitter": 2.0, "alpha": 0.9}}}, [1, 2])
    out, info = texture(base, base, gl, p, seed=3)
    assert np.array_equal(out[:, :80], base[:, :80])                 # skipped group: bit-identical
    assert (out[:, 80:] != base[:, 80:]).any()                        # textured group changed
    d_right = np.linalg.norm(rgb_to_lab(out)[:, 80:] - rgb_to_lab(base)[:, 80:], axis=-1)
    assert d_right.max() < 12                                          # nothing from the other group (dE 25)
    shift = np.linalg.norm(rgb_to_lab(out)[:, 80:].mean((0, 1)) - rgb_to_lab(base)[:, 80:].mean((0, 1)))
    assert shift < 0.5, shift                                          # zero-mean jitter keeps the colour

def test_deterministic():
    base = flat_img(); gl = np.ones(base.shape[:2], np.int32); p = load_director(None, [1])
    o1, _ = texture(base, base, gl, p, seed=5)
    o2, _ = texture(base, base, gl, p, seed=5)
    o3, _ = texture(base, base, gl, p, seed=6)
    assert np.array_equal(o1, o2) and not np.array_equal(o1, o3)

def test_director_validation():
    ok = load_director({"_note": "x", "groups": {"1": {"_c": 1, "size": 5}}}, [1, 2])
    assert ok[1]["size"] == 5 and ok[2]["size"] == DEFAULTS["size"]
    for bad in ({"groups": {"1": {"sizee": 5}}}, {"groups": {"1": {"size": 500}}},
                {"groups": {"9": {}}}, {"defaults": {"direction": "left"}},
                {"groups": {"1": {"skip": 1}}}):
        with pytest.raises(ValueError):
            load_director(bad, [1, 2])

import numpy as np
from nsp.flatten import rgb_to_lab, lab_to_rgb
from nsp.edges import edge_blend

def stripes():
    # A | B | C vertical stripes. A-B dE=3 (soft), B-C dE=10 (hard), flat inside each.
    H, W = 60, 90
    cols = [(50, 0, 0), (53, 0, 0), (63, 0, 0)]
    lab = np.zeros((H, W, 3)); labels = np.zeros((H, W), int)
    for i, c in enumerate(cols):
        x0, x1 = i * 30, (i + 1) * 30
        lab[:, x0:x1] = c; labels[:, x0:x1] = i + 1
    return lab_to_rgb(lab), labels

def jump(rgb, x):
    lab = rgb_to_lab(rgb)
    return np.linalg.norm(lab[:, x] - lab[:, x - 1], axis=-1).mean()

def test_soft_edge_smudged_hard_edge_kept():
    rgb, labels = stripes()
    out, st = edge_blend(rgb, labels)
    lab_in, lab_out = rgb_to_lab(rgb), rgb_to_lab(out)
    # soft boundary (A|B at x=30): jump shrinks
    assert jump(out, 30) < jump(rgb, 30) * 0.6
    # hard boundary (B|C at x=60): untouched within round-trip error
    assert np.linalg.norm(lab_out[:, 58:63] - lab_in[:, 58:63], axis=-1).max() < 0.5
    # interior far from any edge untouched
    assert np.linalg.norm(lab_out[:, 5:20] - lab_in[:, 5:20], axis=-1).max() < 0.5
    assert st["soft_px"] > 0 and st["hard_px"] > 0 and st["changed_px"] > 0

def test_no_edges_is_identity():
    rgb = np.full((20, 20, 3), 120, np.uint8)
    labels = np.ones((20, 20), int)
    out, st = edge_blend(rgb, labels)
    assert np.array_equal(out, rgb) and st["changed_px"] == 0

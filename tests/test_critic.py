import numpy as np
from scipy import ndimage as ndi
from nsp.flatten import rgb_to_lab, lab_to_rgb
from nsp.critic import grid_critic
from tests.synth import make

def base():
    rgb, clean, gt = make(white=0.0, blotch=0.0)   # clean synthetic, labels = ground truth
    return rgb, gt.astype(np.int32)

def cell_flags(rep, cid):
    return next(c["flags"] for c in rep["cells"] if c["id"] == cid)

def test_identical_output_flags_nothing():
    rgb, labels = base()
    rep = grid_critic(rgb, rgb, labels, cell=64)
    assert rep["flagged"] == []

def test_residual_noise_flagged_in_that_cell_only():
    rgb, labels = base()
    lab = rgb_to_lab(rgb)
    rng = np.random.default_rng(1)
    lab[0:64, 0:64, 0] += rng.normal(0, 3.0, (64, 64))     # speckle in the top-left flat sky
    out = lab_to_rgb(lab)
    rep = grid_critic(rgb, out, labels, cell=64)
    assert "residual" in cell_flags(rep, "r0c0")
    assert "r1c3" not in rep["flagged"]

def test_colour_drift_flagged_as_fidelity():
    rgb, labels = base()
    lab = rgb_to_lab(rgb)
    lab[128:192, 320:384] += np.array([8.0, 0, 0])          # visible shift in one cell
    rep = grid_critic(rgb, lab_to_rgb(lab), labels, cell=64)
    assert "fidelity" in cell_flags(rep, "r2c5")

def test_lost_edge_flagged_as_missing():
    rgb, labels = base()
    lab = rgb_to_lab(rgb)
    # blur across the sun edge in one cell: source edge gradient collapses in output
    sub = lab[60:130, 500:620]
    lab[60:130, 500:620] = np.stack([ndi.gaussian_filter(sub[..., c], 6) for c in range(3)], -1)
    rep = grid_critic(rgb, lab_to_rgb(lab), labels, cell=64)
    assert "missing" in cell_flags(rep, "r1c8")     # sun edge sits in cell r1c8

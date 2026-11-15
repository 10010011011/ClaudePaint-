import numpy as np
from scipy import ndimage as ndi
from nsp.flatten import flatten, rgb_to_lab
from tests.synth import make

def err_map(a, b):
    return np.linalg.norm(rgb_to_lab(a) - b, axis=-1)

def purity(labels, gt):
    gt = np.where(gt == 6, 5, gt)   # GT 5|6 differ by dE~1.5: invisible, merging them is correct
    return sum(np.bincount(gt[labels == i]).max() for i in np.unique(labels)) / labels.size

def edge_band(gt, r=3):
    g = gt.astype(float)
    e = (np.abs(np.gradient(g)[0]) + np.abs(np.gradient(g)[1])) > 0
    return ndi.binary_dilation(e, iterations=r)

def run(white, blotch, tol):
    rgb, clean, gt = make(white=white, blotch=blotch)
    r = flatten(rgb, tol=tol)
    en, ef, band = err_map(rgb, clean), err_map(r.flat_rgb, clean), edge_band(gt)
    info = dict(noise_rms=round(float(np.sqrt(3 * (white**2 + blotch**2))), 2), tol=tol,
                regions=r.stats["regions"], after_merge=r.stats["regions_after_merge"],
                grad=r.stats["gradient_regions"],
                colors=(r.stats["perceptual_colors_in"], r.stats["perceptual_colors_out"]),
                dE_noisy=round(float(en.mean()), 2), dE_flat=round(float(ef.mean()), 2),
                edge_noisy=round(float(en[band].mean()), 2), edge_flat=round(float(ef[band].mean()), 2),
                purity=round(purity(r.labels, gt), 4), sec=r.stats["seconds"])
    print(info)
    return r, gt, info

def test_subtle_noise():       # noise RMS below tol: flatten must clean it up
    r, gt, i = run(0.5, 0.8, 2.0)
    assert i["dE_flat"] < i["dE_noisy"] * 0.8
    assert i["edge_flat"] < 3.0                      # edges stay where they were
    assert i["purity"] > 0.995
    assert i["colors"][1] < i["colors"][0] / 3
    maj = lambda k: np.bincount(r.labels[gt == k]).argmax()
    assert maj(5) == maj(6)    # dE~1.5 pair is invisible -> merged
    assert maj(7) != maj(8)    # dE~7 pair is a real edge -> kept
    assert len(np.unique(r.labels[gt == 1])) <= 12   # sky gradient not banded

def test_strong_noise_needs_higher_tol():   # tol must exceed noise RMS (documented behaviour)
    _, _, lo = run(1.2, 1.5, 2.0)
    _, _, hi = run(1.2, 1.5, 4.5)
    assert hi["regions"] < lo["regions"] / 3
    assert hi["dE_flat"] < hi["dE_noisy"] * 0.8

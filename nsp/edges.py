"""Step 2: lost-and-found edges.

Boundaries between two regions whose colours are close (dE below soft_lo) get a narrow
smudged band (Lab Gaussian mix). Boundaries with a visible contrast (dE above soft_hi)
stay crisp. In between, the blend fades linearly. Each boundary pixel takes the MIN
softness over its differing neighbours, so a pixel touching any hard edge stays crisp.
"""
import numpy as np
from scipy import ndimage as ndi
from .flatten import rgb_to_lab, lab_to_rgb

def edge_blend(rgb, labels, soft_lo=3.0, soft_hi=6.0, width=2, sigma=1.2):
    lab = rgb_to_lab(rgb)
    H, W, _ = lab.shape
    m = int(labels.max())
    flat_lbl = labels.ravel()
    cnt = np.maximum(np.bincount(flat_lbl, minlength=m + 1), 1).astype(float)
    mean = np.stack([np.bincount(flat_lbl, weights=lab[..., c].ravel(), minlength=m + 1)
                     for c in range(3)], 1) / cnt[:, None]

    P = np.pad(labels, 1, mode="edge")
    nbs = [P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]]
    bnd = np.zeros((H, W), bool)
    s_pix = np.ones((H, W))
    for nb in nbs:
        diff = nb != labels
        d = np.linalg.norm(mean[labels] - mean[nb], axis=-1)
        s = np.clip((soft_hi - d) / (soft_hi - soft_lo), 0.0, 1.0)
        s_pix = np.where(diff, np.minimum(s_pix, s), s_pix)
        bnd |= diff
    if not bnd.any():
        return rgb.copy(), dict(boundary_px=0, soft_px=0, hard_px=0, changed_px=0)

    # propagate each boundary pixel's softness into a band of `width` px around it
    dist, (iy, ix) = ndi.distance_transform_edt(~bnd, return_indices=True)
    S = s_pix[iy, ix]
    band = (dist <= width) & (S > 0)
    alpha = np.where(band, S * (1.0 - dist / (width + 1.0)), 0.0)

    blur = np.stack([ndi.gaussian_filter(lab[..., c], sigma) for c in range(3)], -1)
    out_lab = lab * (1 - alpha[..., None]) + blur * alpha[..., None]
    out = lab_to_rgb(out_lab)

    changed = np.abs(out.astype(int) - rgb.astype(int)).max(-1) > 0
    stats = dict(boundary_px=int(bnd.sum()),
                 soft_px=int((bnd & (s_pix > 0.5)).sum()),
                 hard_px=int((bnd & (s_pix <= 0.5)).sum()),
                 changed_px=int(changed.sum()))
    return out, stats

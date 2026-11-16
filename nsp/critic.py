"""Step 3: grid critic.

Splits the output into cells and scores each one against the source image. Only flagged
cells go back to the director, as small crops, so token cost scales with problems, not
canvas size.

Metrics per cell (all in Lab / per-pixel units):
  fidelity   mean dE(output, source)           -> colour drifted too far
  residual   rms of output high-pass (L)        -> leftover diffusion noise in flat areas
  missing    frac of strong source edges where the output gradient collapsed -> edge lost
  spurious   frac of output boundaries on source-flat pixels -> invented seam
Thresholds: fidelity calibrated on 2 real images, the rest still guesses.
"""
import json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from .flatten import rgb_to_lab

# fidelity 5.0: calibrated on 2 real diffusion images (2.5 flagged ~half of all cells; flatten itself
# changes real art by ~2 dE). residual 0.8 never fired on them (their noise is ~0.4 dE).
DEFAULT_THR = dict(fidelity=5.0, residual=0.8, missing=0.25, spurious=0.25)

def _grad(L):
    gx = ndi.gaussian_filter(L, 1.0, order=(0, 1))
    gy = ndi.gaussian_filter(L, 1.0, order=(1, 0))
    return np.hypot(gx, gy)

def grid_critic(src_rgb, out_rgb, labels, cell=64, thr=None):
    t = dict(DEFAULT_THR, **(thr or {}))
    ls, lo = rgb_to_lab(src_rgb), rgb_to_lab(out_rgb)
    H, W = labels.shape
    gs, go = _grad(ls[..., 0]), _grad(lo[..., 0])
    src_edge = gs > 3.0
    src_flat = ndi.maximum_filter(gs, 3) < 1.0
    P = np.pad(labels, 1, mode="edge")
    out_bnd = np.zeros((H, W), bool)
    for nb in (P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]):
        out_bnd |= nb != labels
    fid = np.linalg.norm(lo - ls, axis=-1)
    hp = lo[..., 0] - ndi.gaussian_filter(lo[..., 0], 2.0)
    # noise only counts in flat interiors: a high-pass responds to every real step
    near_edge = ndi.binary_dilation(out_bnd | src_edge, iterations=3)
    flat_px = ~near_edge
    missing_px = src_edge & (go < 0.4 * gs)
    spurious_px = out_bnd & src_flat

    rows, cols = -(-H // cell), -(-W // cell)
    cells = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell, min((r + 1) * cell, H)
            x0, x1 = c * cell, min((c + 1) * cell, W)
            sl = (slice(y0, y1), slice(x0, x1))
            n = (y1 - y0) * (x1 - x0)
            m = dict(
                fidelity=float(fid[sl].mean()),
                residual=float(np.sqrt((hp[sl][flat_px[sl]] ** 2).mean())) if flat_px[sl].sum() > 50 else 0.0,
                missing=float(missing_px[sl].sum() / max(src_edge[sl].sum(), 1)),
                spurious=float(spurious_px[sl].sum() / max(n, 1)))
            flags = [k for k in ("fidelity", "residual", "missing", "spurious") if m[k] > t[k]]
            if m["missing"] > 0 and src_edge[sl].sum() < 5:   # too few edge px to judge
                flags = [f for f in flags if f != "missing"]
            cells.append(dict(id=f"r{r}c{c}", row=r, col=c, bbox=[x0, y0, x1, y1],
                              metrics={k: round(v, 3) for k, v in m.items()}, flags=flags))
    flagged = [x["id"] for x in cells if x["flags"]]
    return dict(size=[W, H], cell=cell, grid=[rows, cols], thresholds=t,
                flagged=flagged, cells=cells)

def critic_overlay(out_rgb, report):
    img = Image.fromarray(out_rgb.copy())
    d = ImageDraw.Draw(img)
    for c in report["cells"]:
        x0, y0, x1, y1 = c["bbox"]
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(90, 90, 90))
        if c["flags"]:
            d.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], outline=(255, 40, 40), width=2)
            d.text((x0 + 3, y0 + 2), c["id"], fill=(255, 255, 0))
    return np.array(img)

def write_crops(src_rgb, out_rgb, report, folder, pad=16):
    import os
    os.makedirs(folder, exist_ok=True)
    H, W = out_rgb.shape[:2]
    for cid in report["flagged"]:
        c = next(x for x in report["cells"] if x["id"] == cid)
        x0, y0, x1, y1 = c["bbox"]
        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
        a = src_rgb[Y0:Y1, X0:X1]; b = out_rgb[Y0:Y1, X0:X1]
        pair = np.concatenate([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b], 1)
        Image.fromarray(pair).save(f"{folder}/{cid}.png")

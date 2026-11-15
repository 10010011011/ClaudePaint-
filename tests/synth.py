"""Synthetic 'diffusion-like' test image with known ground truth regions."""
import numpy as np
from scipy import ndimage as ndi
from nsp.flatten import lab_to_rgb

def make(W=768, H=512, seed=0, white=1.0, blotch=1.5):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W]
    lab = np.zeros((H, W, 3)); gt = np.zeros((H, W), int)
    # 1 sky: vertical gradient (blue -> warm), top 60%
    t = np.clip(ys / (0.6 * H), 0, 1)
    sky = (ys < 0.6 * H)
    lab[sky] = np.stack([70 + 12 * t, -4 + 20 * t, -28 + 40 * t], -1)[sky]; gt[sky] = 1
    # 2 ground: flat
    g = ~sky; lab[g] = (35, 12, 18); gt[g] = 2
    # 3 mountain polygon: flat dark
    mtn = (ys > 0.6 * H - 120 + np.abs(xs - 250) * 0.5) & (ys < 0.6 * H + 2) & sky
    lab[mtn] = (28, 4, -6); gt[mtn] = 3
    # 4 sun disc: flat
    sun = (xs - 560) ** 2 + (ys - 120) ** 2 < 45 ** 2
    lab[sun] = (85, 5, 45); gt[sun] = 4
    # 5/6 near-shade pair (dE ~1.5, should merge) ; 7/8 clear pair (dE ~7, must stay split)
    r5 = (xs > 420) & (xs < 480) & (ys > 330) & (ys < 470); lab[r5] = (55, 30, 20); gt[r5] = 5
    r6 = (xs >= 480) & (xs < 540) & (ys > 330) & (ys < 470); lab[r6] = (56, 30.8, 20.9); gt[r6] = 6
    r7 = (xs > 580) & (xs < 640) & (ys > 330) & (ys < 470); lab[r7] = (60, -20, 10); gt[r7] = 7
    r8 = (xs >= 640) & (xs < 700) & (ys > 330) & (ys < 470); lab[r8] = (67, -20, 10); gt[r8] = 8
    clean = lab.copy()
    noise = rng.normal(0, white, lab.shape)                       # speckle
    low = np.stack([ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 25) for _ in range(3)], -1)
    low = low / low.std() * blotch                                # low-frequency blotches
    noisy = lab + noise + low
    return lab_to_rgb(noisy), clean, gt

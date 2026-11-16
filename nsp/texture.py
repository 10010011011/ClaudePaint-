"""Step 4: stroke texture, driven by a small director JSON.

Strokes are traced along the source's own flow (structure tensor of L, i.e. along isophotes)
or along a fixed angle, coloured from the base render at their seed plus a zero-mean Lab
jitter, and clipped to the group that owns them: strokes never recolour a neighbouring group,
so edges stay put. Same seed -> same pixels.

Director JSON (strict: unknown fields raise; keys starting with "_" are ignored):
  {"defaults": {...}, "groups": {"18": {"size": 6, "jitter": 1.0, "direction": "flow"}, "1": {"skip": true}}}
Fields: size px (3-80), jitter dE (0-8), alpha (0-1), length x size (0.5-4), density (0.3-3),
        direction "flow" or degrees (0 = right, 90 = up), skip (bool).
"""
import json, math
import numpy as np
import cv2
from scipy import ndimage as ndi
from .flatten import rgb_to_lab, lab_to_rgb

DEFAULTS = dict(size=10.0, jitter=1.2, alpha=0.7, length=2.4, density=1.0, direction="flow", skip=False)
RANGES = dict(size=(3, 80), jitter=(0, 8), alpha=(0, 1), length=(0.5, 4), density=(0.3, 3))

def _check(params, where):
    for k, v in params.items():
        if k.startswith("_"):
            continue
        if k not in DEFAULTS:
            raise ValueError(f"{where}: unknown field '{k}' (allowed: {sorted(DEFAULTS)})")
        if k == "skip":
            if not isinstance(v, bool):
                raise ValueError(f"{where}: skip must be true/false")
        elif k == "direction":
            if v != "flow" and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise ValueError(f"{where}: direction must be \"flow\" or degrees")
        else:
            lo, hi = RANGES[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{where}: {k}={v!r} outside [{lo}, {hi}]")

def load_director(src, group_ids):
    obj = {} if src is None else (src if isinstance(src, dict) else json.load(open(src)))
    base = dict(DEFAULTS)
    _check(obj.get("defaults", {}), "defaults")
    base.update({k: v for k, v in obj.get("defaults", {}).items() if not k.startswith("_")})
    out = {int(g): dict(base) for g in group_ids}
    for gid, p in obj.get("groups", {}).items():
        if gid.startswith("_"):
            continue
        if int(gid) not in out:
            raise ValueError(f"groups: unknown group id {gid}")
        _check(p, f"group {gid}")
        out[int(gid)].update({k: v for k, v in p.items() if not k.startswith("_")})
    return out

def orientation(L, gl, sigma=1.5, rho=8.0):
    """Stroke direction (radians, image coords) along isophotes; weak-structure pixels inherit the
    group's mean orientation (doubled-angle averaging)."""
    gx = ndi.gaussian_filter(L, sigma, order=(0, 1))
    gy = ndi.gaussian_filter(L, sigma, order=(1, 0))
    jxx, jyy, jxy = (ndi.gaussian_filter(a, rho) for a in (gx * gx, gy * gy, gx * gy))
    zr, zi = jxx - jyy, 2 * jxy
    n = int(gl.max()) + 1
    f = gl.ravel()
    gzr = np.bincount(f, weights=zr.ravel(), minlength=n)
    gzi = np.bincount(f, weights=zi.ravel(), minlength=n)
    mag = np.hypot(gzr, gzi)
    ur = np.where(mag > 1e-9, gzr / np.maximum(mag, 1e-9), 1.0)[gl]
    ui = np.where(mag > 1e-9, gzi / np.maximum(mag, 1e-9), 0.0)[gl]
    floor = 0.05 * (jxx + jyy).mean() + 1e-6
    theta_grad = 0.5 * np.arctan2(zi + floor * ui, zr + floor * ur)
    return theta_grad + np.pi / 2

def _trace(sx, sy, sign, nh, ds, theta_at, gl, g, steps):
    H, W = gl.shape
    x, y = sx.copy(), sy.copy()
    alive = np.ones(len(x), bool)
    th = theta_at(x, y)
    dx, dy = np.cos(th) * sign, np.sin(th) * sign
    X, Y = [x.copy()], [y.copy()]
    for k in range(nh):
        th = theta_at(x, y)
        cx, cy = np.cos(th), np.sin(th)
        flip = (cx * dx + cy * dy) < 0
        dx, dy = np.where(flip, -cx, cx), np.where(flip, -cy, cy)
        nx, ny = x + ds * dx, y + ds * dy
        ix, iy = np.round(nx).astype(int), np.round(ny).astype(int)
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ok = inside.copy()
        ok[inside] = gl[iy[inside], ix[inside]] == g
        alive &= ok & (k < steps)
        x, y = np.where(alive, nx, x), np.where(alive, ny, y)
        X.append(x.copy()); Y.append(y.copy())
    return np.stack(X, 1), np.stack(Y, 1)

def texture(base_rgb, src_rgb, gl, params, seed=0, ds=1.5):
    rng = np.random.default_rng(seed)
    H, W = gl.shape
    theta_flow = orientation(rgb_to_lab(src_rgb)[..., 0], gl)
    base_lab = rgb_to_lab(base_rgb)
    slices = ndi.find_objects(gl)
    S = dict(X=[], Y=[], rgb=[], w=[], gid=[])
    per_group = {}
    for g in range(1, int(gl.max()) + 1):
        p = params.get(g)
        if p is None or p["skip"] or p["alpha"] <= 0 or slices[g - 1] is None:
            continue
        sl = slices[g - 1]
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        step = max(2.0, p["size"] * 0.8 / math.sqrt(p["density"]))
        gy, gx = np.meshgrid(np.arange(y0, y1, step), np.arange(x0, x1, step), indexing="ij")
        iy = np.clip(np.round(gy + rng.uniform(-.4, .4, gy.shape) * step), 0, H - 1).astype(int).ravel()
        ix = np.clip(np.round(gx + rng.uniform(-.4, .4, gx.shape) * step), 0, W - 1).astype(int).ravel()
        keep = gl[iy, ix] == g
        iy, ix = iy[keep], ix[keep]
        n = len(iy)
        if n == 0:
            continue
        length = p["size"] * p["length"] * rng.uniform(.7, 1.3, n)
        steps = length / (2 * ds)
        nh = max(1, int(math.ceil(steps.max())))
        if p["direction"] == "flow":
            theta_at = lambda x, y: theta_flow[np.clip(np.round(y).astype(int), 0, H - 1),
                                               np.clip(np.round(x).astype(int), 0, W - 1)]
        else:
            th0 = np.full(n, -math.radians(p["direction"]))
            theta_at = lambda x, y, th0=th0: th0
        fx, fy = ix.astype(float), iy.astype(float)
        bX, bY = _trace(fx, fy, -1.0, nh, ds, theta_at, gl, g, steps)
        fX, fY = _trace(fx, fy, 1.0, nh, ds, theta_at, gl, g, steps)
        X = np.concatenate([bX[:, :0:-1], fX], 1)
        Y = np.concatenate([bY[:, :0:-1], fY], 1)
        lab = base_lab[iy, ix].copy()
        lab[:, 0] += rng.normal(0, p["jitter"], n)
        lab[:, 1:] += rng.normal(0, p["jitter"] * 0.4, (n, 2))
        S["X"].append(X); S["Y"].append(Y)
        S["rgb"].append(lab_to_rgb(lab[:, None, :])[:, 0, :])
        S["w"].append(np.maximum(1, np.round(p["size"] * 0.55 * rng.uniform(.8, 1.2, n))).astype(int))
        S["gid"].append(np.full(n, g))
        per_group[g] = n
    if not S["X"]:
        return base_rgb.copy(), dict(strokes=0, per_group={})
    cat = {k: np.concatenate(v, 0) for k, v in S.items()}
    N = len(cat["gid"])
    layer = np.zeros((H, W, 3), np.uint8)
    owner = np.zeros((H, W), np.uint16)
    for i in rng.permutation(N):
        pts = np.stack([cat["X"][i], cat["Y"][i]], 1).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer, [pts], False, tuple(int(c) for c in cat["rgb"][i]), int(cat["w"][i]), cv2.LINE_8)
        cv2.polylines(owner, [pts], False, int(cat["gid"][i]), int(cat["w"][i]), cv2.LINE_8)
    out = base_rgb.astype(np.float32).copy()
    layer_f = layer.astype(np.float32)
    for g in per_group:
        sl = slices[g - 1]
        crop = (slice(sl[0].start, sl[0].stop), slice(sl[1].start, sl[1].stop))
        in_g = gl[crop] == g
        valid = in_g & (owner[crop] == g)
        vf = valid.astype(np.float32)
        num = np.stack([cv2.GaussianBlur(layer_f[crop][..., c] * vf, (0, 0), 0.8) for c in range(3)], -1)
        den = cv2.GaussianBlur(vf, (0, 0), 0.8)
        col = num / np.maximum(den[..., None], 1e-6)
        a = (params[g]["alpha"] * np.clip(den * 1.5, 0, 1) * in_g)[..., None]
        out[crop] = out[crop] * (1 - a) + col * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), dict(strokes=int(N), per_group=per_group)

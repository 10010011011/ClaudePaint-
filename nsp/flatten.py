"""Step 1: perceptual flatten.

Merge superpixels while a *plane* in Lab space (per channel: v = a + b*x + c*y) still
fits the merged region within `tol` RMS dE. Flat areas become uniform; smooth gradients
become one region with a fitted gradient (no banding); real edges / texture stay split.
Merging uses mergeable sufficient statistics, so every cost check is O(1).
"""
from __future__ import annotations
import heapq, time
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage import color, segmentation

# ---------------------------------------------------------------- colour helpers
def rgb_to_lab(rgb_u8):
    return color.rgb2lab(rgb_u8)

def lab_to_rgb(lab):
    return (np.clip(color.lab2rgb(lab), 0, 1) * 255 + 0.5).astype(np.uint8)

def perceptual_color_count(lab, cell=2.3):
    """Distinct colours after snapping to ~1 JND cells (dE ~2.3). Metric, not a filter."""
    q = np.round(lab / cell).astype(np.int64).reshape(-1, 3) + 1000
    return int(np.unique(q[:, 0] * 4_000_000 + q[:, 1] * 2000 + q[:, 2]).size)

# ---------------------------------------------------------------- statistics
# per region: [n,Sx,Sy,Sxx,Sxy,Syy] + per channel c: [Sv,Sxv,Syv,Svv]  (18 floats)
def _stats(lab, labels, n):
    H, W, _ = lab.shape
    ys, xs = np.mgrid[0:H, 0:W]
    x = (xs / max(W - 1, 1)).ravel()
    y = (ys / max(H - 1, 1)).ravel()
    lb = labels.ravel()
    bc = lambda w=None: np.bincount(lb, weights=w, minlength=n)
    S = np.zeros((n, 18))
    S[:, 0], S[:, 1], S[:, 2] = bc(), bc(x), bc(y)
    S[:, 3], S[:, 4], S[:, 5] = bc(x * x), bc(x * y), bc(y * y)
    for c in range(3):
        v = lab[..., c].ravel()
        b = 6 + 4 * c
        S[:, b], S[:, b + 1], S[:, b + 2], S[:, b + 3] = bc(v), bc(x * v), bc(y * v), bc(v * v)
    return S

def _fit(s):
    """Least-squares plane per Lab channel from stats -> beta (3 coeff x 3 ch), sse (3,)."""
    n = s[0]
    A = np.array([[n, s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]])
    V = s[6:].reshape(3, 4)
    rhs = V[:, :3].T
    beta = np.linalg.solve(A + np.eye(3) * 1e-9 * max(n, 1.0), rhs)
    sse = V[:, 3] - (beta * rhs).sum(0)
    return beta, np.maximum(sse, 0.0)

def _cost(s):
    if s[0] <= 3:
        return 0.0
    return float(np.sqrt(_fit(s)[1].sum() / s[0]))

def _adjacency(lab, labels, n):
    """Neighbour sets + per-pair boundary stats: key (i<j) -> [sum(lab_i - lab_j) over straddling
    pixel pairs (3-vec), count]. Contrast = |mean| -> a plane may not hide a real edge."""
    la = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    lb = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    va = np.concatenate([lab[:, :-1].reshape(-1, 3), lab[:-1, :].reshape(-1, 3)])
    vb = np.concatenate([lab[:, 1:].reshape(-1, 3), lab[1:, :].reshape(-1, 3)])
    m = la != lb
    la, lb, va, vb = la[m], lb[m], va[m], vb[m]
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    d = np.where((la < lb)[:, None], va - vb, vb - va)
    keys, inv, cnt = np.unique(lo.astype(np.int64) * n + hi, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(inv, weights=d[:, c], minlength=len(keys)) for c in range(3)], 1)
    adj = [set() for _ in range(n)]
    edge = {}
    for k, sm, c in zip(keys, sums, cnt):
        i, j = int(k // n), int(k % n)
        adj[i].add(j); adj[j].add(i)
        edge[(i, j)] = [sm, float(c)]
    return adj, edge

def _merge(S, adj, edge, tol, edge_tol):
    n = len(S)
    alive = np.ones(n, bool)
    ver = np.zeros(n, int)
    parent = np.arange(n)

    def contrast(a, b):
        sm, c = edge[(a, b) if a < b else (b, a)]
        return float(np.linalg.norm(sm / c))

    def cost(a, b):
        if contrast(a, b) > edge_tol:      # a visible edge between them: never merge
            return np.inf
        return _cost(S[a] + S[b])

    heap = [(cost(a, b), a, b, 0, 0) for a in range(n) for b in adj[a] if a < b]
    heap = [h for h in heap if h[0] <= tol]
    heapq.heapify(heap)
    while heap:
        c, a, b, va, vb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb:
            continue
        S[a] += S[b]
        alive[b] = False
        ver[a] += 1
        parent[b] = a
        edge.pop((min(a, b), max(a, b)), None)
        for nb in list(adj[b]):
            if nb == a:
                continue
            kb = (b, nb) if b < nb else (nb, b)
            sm, cn = edge.pop(kb)
            if b > nb:                       # stored as nb - b ; want b - nb
                sm = -sm
            ka = (a, nb) if a < nb else (nb, a)
            add = sm if a < nb else -sm      # store as lower-id minus higher-id
            if ka in edge:
                edge[ka][0] = edge[ka][0] + add; edge[ka][1] += cn
            else:
                edge[ka] = [add, cn]
            adj[nb].discard(b); adj[nb].add(a); adj[a].add(nb)
        adj[a].discard(b); adj[a].discard(a)
        adj[b] = set()
        for nb in adj[a]:
            cc = cost(a, nb)
            if cc <= tol:
                heapq.heappush(heap, (cc, a, nb, ver[a], ver[nb]))
    for i in range(n):  # resolve parents
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    return parent

# ---------------------------------------------------------------- models + boundary refinement
def _compact(labels):
    """Re-id regions 1..N, largest area first."""
    ids, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    rank = np.empty(len(ids), np.int32)
    rank[order] = np.arange(1, len(ids) + 1)
    return rank[inv].reshape(labels.shape), counts[order]

def _models(lab, labels, m, grad_span, min_plane_px):
    """Per-region model: constant mean, or a Lab plane where the plane is real.
    Returns betas (m+1, 3 coeff, 3 ch) and use_plane (m+1,)."""
    S = _stats(lab, labels, m + 1)
    betas = np.zeros((m + 1, 3, 3))
    use = np.zeros(m + 1, bool)
    for i in range(1, m + 1):
        n = S[i, 0]
        if n == 0:
            continue
        betas[i, 0] = S[i, [6, 10, 14]] / n
        if n >= min_plane_px:
            b = _fit(S[i])[0]
            vx = max(S[i, 3] / n - (S[i, 1] / n) ** 2, 0)
            vy = max(S[i, 5] / n - (S[i, 2] / n) ** 2, 0)
            span = np.linalg.norm(np.abs(b[1]) * np.sqrt(12 * vx) + np.abs(b[2]) * np.sqrt(12 * vy))
            if span > grad_span:
                betas[i], use[i] = b, True
    return betas, use

def _refine(lab, labels, F, passes, lam, grad_span, min_plane_px):
    """ICM boundary snapping: each boundary pixel joins whichever adjacent region's model
    predicts it best (+ Potts smoothness lam). Checkerboard phases avoid oscillation."""
    H, W, _ = lab.shape
    par = (np.add.outer(np.arange(H), np.arange(W))) & 1
    for _ in range(passes):
        m = int(labels.max())
        betas, _ = _models(lab, labels, m, grad_span, min_plane_px)
        changed = 0
        for phase in (0, 1):
            P = np.pad(labels, 1, mode="edge")
            nb = np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]], -1)
            ys, xs = np.nonzero((nb != labels[..., None]).any(-1) & (par == phase))
            if len(ys) == 0:
                continue
            cur = labels[ys, xs]
            nbp = nb[ys, xs]
            cand = np.concatenate([cur[:, None], nbp], 1)                      # n,5
            pred = np.einsum("nk,nskc->nsc", F[ys, xs], betas[cand])           # n,5,3
            cost = ((pred - lab[ys, xs][:, None, :]) ** 2).sum(-1)
            cost += lam * (nbp[:, None, :] != cand[:, :, None]).sum(-1)
            new = cand[np.arange(len(ys)), cost.argmin(1)]
            ch = new != cur
            labels[ys[ch], xs[ch]] = new[ch]
            changed += int(ch.sum())
        if changed == 0:
            break
    return labels

# ---------------------------------------------------------------- main entry
@dataclass
class FlattenResult:
    flat_rgb: np.ndarray      # HxWx3 uint8
    labels: np.ndarray        # HxW int32, ids 1..N, id 1 = largest region
    regions: list             # dicts: id, area, bbox, centroid, mean_hex, gradient, span_dE
    stats: dict

def flatten(rgb, tol=2.0, n_segments=2500, compactness=8.0, sigma=1.0, grad_span=4.0,
            min_plane_px=200, refine_passes=6, smooth=3.0, edge_tol=None):
    t0 = time.time()
    lab = rgb_to_lab(rgb)
    H, W, _ = lab.shape
    sp = segmentation.slic(lab, n_segments=n_segments, compactness=compactness, sigma=sigma,
                           convert2lab=False, start_label=0, channel_axis=-1)
    ys, xs = np.mgrid[0:H, 0:W]
    F = np.stack([np.ones((H, W)), xs / max(W - 1, 1), ys / max(H - 1, 1)], -1)
    # snap superpixel borders to real edges BEFORE merging (SLIC cells straddle steps, and a
    # merge can never undo that); constant models only at this scale.
    sp = _refine(lab, sp + 1, F, 4, smooth, grad_span, 10 ** 9)
    sp = _compact(sp)[0] - 1
    n0 = int(sp.max()) + 1
    S = _stats(lab, sp, n0)
    adj, edge = _adjacency(lab, sp, n0)
    parent = _merge(S, adj, edge, tol, 1.5 * tol if edge_tol is None else edge_tol)
    labels, _ = _compact(parent[sp])
    n_merged = int(labels.max())
    labels = _refine(lab, labels, F, refine_passes, smooth, grad_span, min_plane_px)
    labels, counts = _compact(labels)
    m = int(labels.max())

    betas, use_plane = _models(lab, labels, m, grad_span, min_plane_px)
    pred = np.einsum("hwk,hwkc->hwc", F, betas[labels])
    idx = np.arange(1, m + 1)
    lo = np.stack([ndi.minimum(lab[..., c], labels, idx) for c in range(3)], 1)
    hi = np.stack([ndi.maximum(lab[..., c], labels, idx) for c in range(3)], 1)
    pred = np.clip(pred, np.vstack([lo[:1], lo])[labels], np.vstack([hi[:1], hi])[labels])
    flat_rgb = lab_to_rgb(pred)
    pmin = np.stack([ndi.minimum(pred[..., c], labels, idx) for c in range(3)], 1)
    pmax = np.stack([ndi.maximum(pred[..., c], labels, idx) for c in range(3)], 1)
    span = np.linalg.norm(pmax - pmin, axis=1)
    mean_lab = np.stack([ndi.mean(pred[..., c], labels, idx) for c in range(3)], 1)
    mean_hex = ["#%02x%02x%02x" % tuple(lab_to_rgb(mean_lab[i][None, None])[0, 0]) for i in range(m)]
    slices = ndi.find_objects(labels, max_label=m)
    cy = ndi.center_of_mass(np.ones_like(labels), labels, idx)
    regions = []
    for i in range(m):
        sl = slices[i]
        regions.append(dict(
            id=i + 1, area=int(counts[i]), area_frac=round(float(counts[i]) / (H * W), 5),
            bbox=[int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
            centroid=[round(cy[i][1], 1), round(cy[i][0], 1)],
            mean_hex=mean_hex[i], span_dE=round(float(span[i]), 2), gradient=bool(use_plane[i + 1])))
    stats = dict(
        size=[W, H], tol=tol, superpixels=n0, regions_after_merge=n_merged, regions=m,
        gradient_regions=int(use_plane.sum()),
        perceptual_colors_in=perceptual_color_count(lab),
        perceptual_colors_out=perceptual_color_count(rgb_to_lab(flat_rgb)),
        mean_dE_flat_vs_input=round(float(np.linalg.norm(rgb_to_lab(flat_rgb) - lab, axis=-1).mean()), 3),
        seconds=round(time.time() - t0, 2))
    return FlattenResult(flat_rgb, labels, regions, stats)

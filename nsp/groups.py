"""Director groups: coarsen thousands of fine regions to ~K numbered groups.

The fine map is for rendering; the director (LLM) can only reason about a few dozen
labelled areas. Near-identical colours first join across gaps (background islands), then the smallest
region repeatedly merges into its most similar neighbour (mean-Lab distance, slight preference
for long shared borders). Groups can therefore be disconnected (e.g. one background group). Every fine region lands
wholly inside one group, so a decision made on a group applies to all its members.
"""
import heapq
import numpy as np
from .flatten import rgb_to_lab, lab_to_rgb

def coarsen(labels, flat_rgb, K=60):
    lab = rgb_to_lab(flat_rgb)
    m = int(labels.max())
    lb = labels.ravel()
    cnt = np.bincount(lb, minlength=m + 1).astype(float)
    mean = np.stack([np.bincount(lb, weights=lab[..., c].ravel(), minlength=m + 1)
                     for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    d = a != b
    keys, c = np.unique(np.minimum(a[d], b[d]).astype(np.int64) * (m + 1) + np.maximum(a[d], b[d]),
                        return_counts=True)
    adj = [dict() for _ in range(m + 1)]
    for k, n in zip(keys, c):
        i, j = int(k // (m + 1)), int(k % (m + 1))
        adj[i][j] = adj[j][i] = float(n)

    alive = np.ones(m + 1, bool); alive[0] = False
    area = cnt.copy()
    parent = np.arange(m + 1)
    heap = [(area[i], i) for i in range(1, m + 1)]
    heapq.heapify(heap)
    n_alive = m
    def merge(i, j):
        nonlocal n_alive
        mean[j] = (mean[j] * area[j] + mean[i] * area[i]) / (area[i] + area[j])
        area[j] += area[i]
        alive[i] = False; parent[i] = j
        for k, l in adj[i].items():
            if k == j:
                continue
            adj[k].pop(i, None)
            adj[k][j] = adj[k].get(j, 0.0) + l
            adj[j][k] = adj[j].get(k, 0.0) + l
        adj[j].pop(i, None); adj[i] = {}
        n_alive -= 1

    # Phase A: near-identical colours join even across gaps (e.g. background islands split by
    # thin wisps). Tight cell (dE ~1.5) so different skin/shadow tones stay apart. Never below K.
    if n_alive > K:
        cells = {}
        for i in range(1, m + 1):
            cells.setdefault(tuple(np.round(mean[i] / 1.5).astype(int)), []).append(i)
        for members in cells.values():
            if len(members) < 2:
                continue
            j = max(members, key=lambda t: area[t])
            for i in members:
                if i != j and n_alive > K:
                    merge(i, j)
        heap = [(area[i], i) for i in range(1, m + 1) if alive[i]]
        heapq.heapify(heap)

    # Phase B: smallest region joins its most similar neighbour until K remain.
    while n_alive > K and heap:
        ar, i = heapq.heappop(heap)
        if not alive[i] or ar != area[i] or not adj[i]:
            continue
        j = min(adj[i], key=lambda t: np.linalg.norm(mean[i] - mean[t]) - 0.5 * np.log1p(adj[i][t]))
        merge(i, j)
        heapq.heappush(heap, (area[j], j))
    for i in range(1, m + 1):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    roots = np.unique(parent[1:])
    order = roots[np.argsort(-area[roots])]
    rank = np.zeros(m + 1, np.int32)
    rank[order] = np.arange(1, len(order) + 1)
    fine_to_group = rank[parent]; fine_to_group[0] = 0
    glabels = fine_to_group[labels].astype(np.int32)

    H, W = labels.shape
    gm = len(order)
    gcnt = np.bincount(glabels.ravel(), minlength=gm + 1)
    from scipy import ndimage as ndi
    sl = ndi.find_objects(glabels, max_label=gm)
    groups = []
    for g in range(1, gm + 1):
        s = sl[g - 1]
        members = int((fine_to_group[1:] == g).sum())
        groups.append(dict(
            id=g, area=int(gcnt[g]), area_frac=round(float(gcnt[g]) / (H * W), 5),
            bbox=[int(s[1].start), int(s[0].start), int(s[1].stop), int(s[0].stop)],
            mean_hex="#%02x%02x%02x" % tuple(lab_to_rgb(mean[order[g - 1]][None, None])[0, 0]),
            fine_regions=members, gradient=False))
    return glabels, fine_to_group, groups

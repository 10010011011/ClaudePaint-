import numpy as np
from nsp.flatten import flatten
from nsp.groups import coarsen
from tests.synth import make

def test_coarsen_consistent():
    rgb, _, gt = make(white=0.5, blotch=0.8)
    r = flatten(rgb, tol=2.0)                     # 7 fine regions
    gl, f2g, groups = coarsen(r.labels, r.flat_rgb, K=4)
    assert len(groups) == 4 and gl.max() == 4
    assert np.array_equal(gl, f2g[r.labels])                      # exact mapping fine -> group
    assert sum(g["area"] for g in groups) == gl.size              # nothing lost
    assert [g["area"] for g in groups] == sorted([g["area"] for g in groups], reverse=True)
    # biggest fine region (sky) is never swallowed by a small one: it stays group 1
    sky = np.bincount(r.labels[gt == 1]).argmax()
    assert gl[r.labels == sky].min() == 1 and gl[r.labels == sky].max() == 1

def test_k_larger_than_regions_is_noop():
    rgb, _, _ = make(white=0.5, blotch=0.8)
    r = flatten(rgb, tol=2.0)
    gl, f2g, groups = coarsen(r.labels, r.flat_rgb, K=50)
    assert len(groups) == r.stats["regions"]

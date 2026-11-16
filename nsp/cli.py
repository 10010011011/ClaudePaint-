import argparse, json, os
import numpy as np
from PIL import Image
from .flatten import flatten
from .overlay import make_overlay
from .edges import edge_blend
from .critic import grid_critic, critic_overlay, write_crops
from .groups import coarsen

def main(argv=None):
    p = argparse.ArgumentParser(prog="nsp")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flatten", help="perceptual flatten + numbered region map")
    f.add_argument("image"); f.add_argument("--out", default="job")
    f.add_argument("--tol", type=float, default=2.0, help="RMS dE tolerance of plane fit (JND ~2.3)")
    f.add_argument("--segments", type=int, default=2500)
    f.add_argument("--max-side", type=int, default=0, help="downscale longest side first (0=off)")
    b = sub.add_parser("blend", help="step 2: smudge soft edges in a flatten job folder")
    b.add_argument("job"); b.add_argument("--width", type=int, default=2)
    b.add_argument("--sigma", type=float, default=1.2)
    k = sub.add_parser("critic", help="step 3: grid critic on a blended job; crops of flagged cells")
    k.add_argument("job"); k.add_argument("--src", required=True, help="original source image")
    k.add_argument("--cell", type=int, default=64)
    k.add_argument("--use", default="blended.png", help="which output to judge (blended.png or flat.png)")
    g = sub.add_parser("groups", help="coarsen fine regions to ~K director groups + numbered overlay")
    g.add_argument("job"); g.add_argument("--k", type=int, default=60)
    a = p.parse_args(argv)
    if a.cmd == "groups":
        flat = np.array(Image.open(f"{a.job}/flat.png").convert("RGB"))
        labels = np.load(f"{a.job}/labels.npy")
        gl, f2g, groups = coarsen(labels, flat, K=a.k)
        np.save(f"{a.job}/groups.npy", gl); np.save(f"{a.job}/fine_to_group.npy", f2g)
        Image.fromarray(make_overlay(flat, gl, groups, max_labels=a.k, min_frac=0.0)).save(f"{a.job}/groups_overlay.png")
        json.dump(groups, open(f"{a.job}/groups.json", "w"))
        print(json.dumps(dict(groups=len(groups), smallest_frac=min(x["area_frac"] for x in groups))))
        return
    if a.cmd == "critic":
        src = np.array(Image.open(a.src).convert("RGB"))
        out = np.array(Image.open(f"{a.job}/{a.use}").convert("RGB"))
        labels = np.load(f"{a.job}/labels.npy")
        if src.shape != out.shape:
            src = np.array(Image.fromarray(src).resize((out.shape[1], out.shape[0]), Image.LANCZOS))
        rep = grid_critic(src, out, labels, cell=a.cell)
        json.dump(rep, open(f"{a.job}/critic.json", "w"), indent=1)
        Image.fromarray(critic_overlay(out, rep)).save(f"{a.job}/critic_overlay.png")
        write_crops(src, out, rep, f"{a.job}/crops")
        print(json.dumps(dict(grid=rep["grid"], flagged=len(rep["flagged"]), ids=rep["flagged"][:20])))
        return
    if a.cmd == "blend":
        flat = np.array(Image.open(f"{a.job}/flat.png").convert("RGB"))
        labels = np.load(f"{a.job}/labels.npy")
        out, st = edge_blend(flat, labels, width=a.width, sigma=a.sigma)
        Image.fromarray(out).save(f"{a.job}/blended.png")
        print(json.dumps(st))
        return
    img = Image.open(a.image).convert("RGB")
    if a.max_side and max(img.size) > a.max_side:
        s = a.max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    rgb = np.array(img)
    r = flatten(rgb, tol=a.tol, n_segments=a.segments)
    os.makedirs(a.out, exist_ok=True)
    Image.fromarray(r.flat_rgb).save(f"{a.out}/flat.png")
    np.save(f"{a.out}/labels.npy", r.labels)
    Image.fromarray(make_overlay(r.flat_rgb, r.labels, r.regions)).save(f"{a.out}/overlay.png")
    json.dump(r.regions, open(f"{a.out}/regions.json", "w"))
    json.dump(r.stats, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(r.stats))

if __name__ == "__main__":
    main()

import argparse, json, os
import numpy as np
from PIL import Image
from .flatten import flatten
from .overlay import make_overlay

def main(argv=None):
    p = argparse.ArgumentParser(prog="nsp")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flatten", help="perceptual flatten + numbered region map")
    f.add_argument("image"); f.add_argument("--out", default="job")
    f.add_argument("--tol", type=float, default=2.0, help="RMS dE tolerance of plane fit (JND ~2.3)")
    f.add_argument("--segments", type=int, default=2500)
    f.add_argument("--max-side", type=int, default=0, help="downscale longest side first (0=off)")
    a = p.parse_args(argv)
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

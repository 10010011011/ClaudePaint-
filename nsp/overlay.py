"""Numbered region overlay: what the director (LLM) looks at instead of raw pixels."""
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage import segmentation

def make_overlay(flat_rgb, labels, regions, max_labels=60, min_frac=0.003, scale=1.0):
    img = (flat_rgb.astype(np.float32) * 0.8).astype(np.uint8)
    edges = segmentation.find_boundaries(labels, mode="inner")
    img[edges] = (255, 255, 255)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    shown = 0
    for r in regions:  # already sorted by area desc
        if shown >= max_labels or r["area_frac"] < min_frac:
            break
        x0, y0, x1, y1 = r["bbox"]
        mask = labels[y0:y1, x0:x1] == r["id"]
        dt = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        py, px = np.unravel_index(np.argmax(dt), dt.shape)
        pos = (int((x0 + px) * scale), int((y0 + py) * scale))
        txt = str(r["id"]) + ("~" if r["gradient"] else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        org = (pos[0] - tw // 2, pos[1] + th // 2)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        r["label_pos"] = [pos[0] / scale, pos[1] / scale]
        shown += 1
    return img

# ClaudePaint (no-slop-painter)

Deterministic painting tools. The LLM only directs: it reads numbered region maps and writes small JSON decisions. It never issues per-stroke calls.

## Status
- Step 1 (done): perceptual flatten + numbered region map
- Step 2 (done): lost-and-found edge blend
- Step 3 (done): grid critic (flags cells for review, crops only those)
- Planned: 4 stroke texture, 5 pixel mode, 6 skill + MCP wrapper

## Step 1: flatten
    pip install -e . pytest
    python -m nsp.cli flatten input.png --out job --tol 2.0

Outputs in `job/`: `flat.png`, `labels.npy`, `overlay.png` (numbered regions, `~` = gradient), `regions.json`, `stats.json`.

- Merges superpixels while a Lab plane still fits within `--tol` RMS dE. Flat areas become uniform, smooth gradients become one fitted region (no banding), real edges stay split.
- Boundary guard: no merge across a visible edge (`edge_tol` = 1.5 x tol). Superpixel borders are snapped to real edges before merging.
- `tol` must exceed the image's noise RMS, or almost nothing merges. Automatic noise estimation is the next item.

## Step 2: edge blend
    python -m nsp.cli blend job --width 2 --sigma 1.2

Writes `job/blended.png` and prints counts of soft and hard boundary pixels.

- Boundaries with dE below 3 are smudged with a narrow Lab Gaussian band. Above 6 they stay crisp. In between, the blend fades linearly.
- Each boundary pixel takes the strongest-contrast neighbour, so a pixel touching any hard edge stays crisp.
- Synthetic test: soft edge max jump 3.16 -> 0.79; hard edge 9.65 unchanged.
- Flatten already merges soft boundaries it can, so on simple images this step changes little.

## Step 3: grid critic
    python -m nsp.cli critic job --src input.png --cell 64

Writes `job/critic.json` (every cell's metrics and flags), `job/critic_overlay.png` (flagged cells in red), and `job/crops/<cell>.png` (source | output, side by side, flagged cells only).

Flags per cell:
- `fidelity`: mean dE from the source is above 2.5
- `residual`: leftover noise in flat interiors (high-pass rms above 0.8)
- `missing`: a strong source edge whose output gradient collapsed
- `spurious`: an output seam on a source-flat area

Thresholds are initial guesses and aren't calibrated on real images yet. Residual is measured on flat interiors only, so real steps don't count as noise.

## Tests
    python -m pytest -q -s tests

Synthetic ground-truth image (sky gradient, flat ground, mountain, sun, two near-identical pairs, two clearly different pairs):
- dE to clean image: 1.53 -> 0.59 (subtle noise, tol 2.0)
- perceptual colours: 411 -> 56
- Near-identical pair (dE ~1.5) merged; clearly different pair (dE ~7) kept apart.
- Sky gradient covered by 12 or fewer regions, no banding.
- Critic: identical output flags nothing; injected noise, colour drift and a lost edge are each flagged in the right cell.
- Not yet verified: exact per-pixel region areas. Only real diffusion images will test this.
- Synthetic images only so far. No real diffusion output has been tested.

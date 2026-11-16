# ClaudePaint (no-slop-painter)

Deterministic painting tools. The LLM only directs: it reads numbered region maps and writes small JSON decisions. It never issues per-stroke calls.

## Status
- Step 1 (done): perceptual flatten + numbered region map
- Step 2 (done): lost-and-found edge blend
- Step 3 (done): grid critic (flags cells for review, crops only those)
- Step 3b (done): director groups (thousands of fine regions -> ~60 numbered groups)
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
- `fidelity`: mean dE from the source is above 5.0
- `residual`: leftover noise in flat interiors (high-pass rms above 0.8)
- `missing`: a strong source edge whose output gradient collapsed
- `spurious`: an output seam on a source-flat area

Residual is measured on flat interiors only, so real steps don't count as noise.

## Step 3b: director groups
    python -m nsp.cli groups job --k 60

Writes `job/groups.npy`, `job/fine_to_group.npy`, `job/groups.json` and `job/groups_overlay.png` (numbered). The fine map renders; the director only reasons about ~60 groups, and each fine region sits wholly inside one group. Near-identical colours join across gaps first (background islands), then the smallest region merges into its most similar neighbour. Groups may be disconnected.

## Real-image calibration (2 anime-style diffusion images, 922x1536 and 1024x1536)
- Per-pixel noise is tiny (~0.3-0.4 dE, 68-82% of flat pixels have zero residual), so `tol 2.0` is well above it.
- `tol 2.0`: 3113 and 1485 fine regions; perceptual colours 20515 -> 10131 and 6201 -> 2542. About 12 s each at `--segments 6000`.
- `tol 3.5` is too aggressive for this style: on image 2 it flagged 88 cells for lost edges, against 22 at `tol 2.0`. Keep 2.0.
- Critic: fidelity threshold raised 2.5 -> 5.0 (2.5 flagged about half of all cells). Now 8% and 6% of cells flagged. `residual` never fired (max 0.32 vs 0.8).
- Only two images, one art style. Thresholds are not general yet.

## Tests
    python -m pytest -q -s tests

Synthetic ground-truth image (sky gradient, flat ground, mountain, sun, two near-identical pairs, two clearly different pairs):
- dE to clean image: 1.53 -> 0.59 (subtle noise, tol 2.0)
- perceptual colours: 411 -> 56
- Near-identical pair (dE ~1.5) merged; clearly different pair (dE ~7) kept apart.
- Sky gradient covered by 12 or fewer regions, no banding.
- Critic: identical output flags nothing; injected noise, colour drift and a lost edge are each flagged in the right cell.
- Groups: exact fine-to-group mapping, no area lost, largest region stays group 1.
- Not yet verified: exact per-pixel region areas. Only real diffusion images will test this.

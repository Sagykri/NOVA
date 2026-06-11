---
name: nova-pixel-colocalization
description: "Pixel-level channel co-localization: conditional overlap fraction, rotation null, Wilcoxon. Use when understanding, scaffolding, or adapting co-localization notebooks for a new experiment."
model: claude-sonnet-4-6
---

# nova-pixel-colocalization

Reference skill for the pixel-level co-localization analysis used in NOVA microscopy experiments. Covers method, pipeline steps, site-vs-tile distinction, thresholding, interpretation, and how to adapt for a new experiment.

---

## 1. Concept

**Goal:** test whether two fluorescence markers co-localize above spatial chance.

**Core metric — conditional overlap fraction (binary masks only):**

```
fraction_overlap = P(partner+ | anchor+)
                 = |anchor_mask AND partner_mask| / |anchor_mask|
```

- **anchor**: marker whose positive pixels form the denominator (e.g. RANBP17, DCP1A P-bodies).
- **partner**: marker tested for enrichment inside the anchor (e.g. TDP-43, V5).
- Score near 1 → partner strongly co-localizes with anchor; near 0 → weak or absent.
- If the anchor mask is empty the score is `None` (excluded from analysis).

**Null model (rotation test):** rotate the partner image while keeping the anchor fixed, then recompute the same fraction → `fraction_overlap_rotated`. This breaks true spatial alignment but preserves each channel's intensity/texture statistics. Real >> rotated indicates genuine co-localization. The current helpers **average over multiple rotations** (`k=1,2,3`, i.e. 90/180/270°) for a more stable null than a single 90° turn.

**Caveat — the rotation null is lenient.** Rotating the partner can land its signal on background, so any two markers that are both enriched in the cell body show *some* above-null overlap. In the AAT_NOVA_pilot2 control run, all four pairs (positive + negatives) cleared the replicate Wilcoxon at the p-floor (12/12 units real>rotated, p≈2.4e-4). What separated the true positive (pDRP1+TOMM20, Δ≈0.12) from the negatives (Δ≈0.04–0.07) was the **effect magnitude**, not significance. So validate with a **positive-vs-negative contrast**, not the rotation p-value alone.

**When to use this analysis:**
- You have two fluorescence channels in the same image/tile and want to quantify spatial co-localization without relying on intensity correlation (Pearson/Mander's).
- You want a biologically interpretable metric (fraction of anchor pixels that contain partner signal).
- You want a built-in per-image spatial-random baseline without an external reference.

---

## 2. Pipeline steps

1. **List paths** — glob all TIFF site files or NPY tile files for the experiment.
2. **Build dataframes** — parse directory hierarchy into columns (batch, cell_line, panel, condition, rep, marker, site, [tile_index]). Use negative-index path parsing for robustness.
3. **Match anchor & partner** — join on shared keys so each row has both channels from the same field/tile.
4. **Threshold & mask** — apply per-image adaptive threshold to produce binary masks.
5. **Score real fraction** — compute `fraction_overlap` per image/tile.
6. **Score rotated fraction** — rotate partner (avg over k=1,2,3), recompute → `fraction_overlap_rotated`.
7. **Aggregate** — group by (batch, rep, condition) to get per-replicate means; the replicate is the inference unit (tiles are pseudoreplicates).
8. **Wilcoxon test** — paired signed-rank (alternative='greater') comparing real vs rotated. **Primary test runs at the replicate level** (per batch×rep means), not per-tile. Per-tile Wilcoxon is descriptive only (inflated n). Needs ≥2 batches for real inference.
9. **Effect-size report** (optional) — `run_analysis_generate_report` from `cell_profiler.code.cp_effect_size_utils`; compares biological groups (e.g. KD vs control) with batch as fixed or random effect. Only meaningful with ≥2 batches (falls back to OLS if singular).
10. **Reproducibility / CV** — coefficient of variation (SD/Mean across reps) per condition. CV < 0.1 excellent, 0.1–0.2 acceptable, > 0.2 high variability.
11. **Plots** — box/scatter/bar of real vs rotated; mask visualization (anchor img, partner img, anchor mask, partner mask, optional RGB overlay).

---

## 3. Site-level vs tile-level

| | **Site-level (TIFF)** | **Tile-level (NPY)** |
|---|---|---|
| Input | Full preprocessed TIFF, ~1024×1024 | NPY tiles, shape `[n_tiles, 100, 100, 2]` |
| Pixel values | Raw 16-bit → rescale to [0,1] with `rescale_intensity(img, lower=0.5, upper=99.9)` before thresholding | Already rescaled to [0,1] — **no rescaling step** |
| Channels | Separate files per channel; matched by substituting the channel token in filename (e.g. `ch2`→`ch3`) | Channel index 0 = target marker, index 1 = DAPI |
| Anchor read | `cv2.imread(path, cv2.IMREAD_ANYDEPTH)` then rescale | `npy[i, :, :, 0]` |
| Matching keys | (batch, cell_line, panel, condition, rep, filename) | (batch, cell_line, condition, rep, site, tile_index) |
| Tile validity | N/A | Pairing is valid **only when the two markers are different channels of the SAME panel** (same physical image → same nucleus segmentation). Then tile `i` is the same nucleus across markers, `n_tiles` match, and the DAPI channel (index 1) is identical. See the same-panel caveat below. |
| Directory layout | `.../sorted/<batch>/<cell_line>/<panel>/<condition>/<rep>/<marker>/<file>.tiff` | `.../processed/<batch>/<cell_line>/<condition>/<marker>/<file>.npy` — panel & rep encoded in filename (e.g. `rep2_r03c04f07-ch1t1_panelA_...`) |
| Shared util | `sites_paths_to_dataframe`, `match_sites_and_score_fraction_overlap` | `load_tiles_to_dataframe`, `match_tiles_and_score_fraction_overlap` |

### ⚠️ Same-panel requirement (tile-level) — check this FIRST

Tile-level co-localization can **only** pair two markers that are different fluorescence channels of the **same panel** (same physical well/field). Multiplex experiments deliberately split markers across panels (e.g. AAT_NOVA_pilot2: panelA = FK-2/TDP-43/SMI32, panelB = pCaMKIIa/pDRP1/TOMM20, panelC = ATF4/pTDP-43/ATF6). Cross-panel markers come from **different images** → their `site`/`tile_index` do not refer to the same nucleus, so pairing is meaningless.

Before choosing pairs:
1. Extract each marker's panel from the filename (`panel[A-Za-z0-9]+`). Only pairs within one panel are feasible.
2. Verify: a cross-panel "pair" yields **0 shared sites**; a valid within-panel pair shares most sites **and** `np.allclose(anchorNPY[...,1], partnerNPY[...,1])` (identical DAPI) for a shared site.

Consequence for control design: because panels multiplex *distinct* organelles, strong **positive** controls are scarce within a panel (most within-panel pairs are different compartments → negatives). Pick the one biologically-linked within-panel pair as the positive (e.g. TOMM20 + pDRP1, both mitochondrial) and draw negatives from across panels.

---

## 4. Thresholding

**Formula:**
```
threshold = mean(img) + K * std(img)
mask = img > threshold
```

K is set per marker. Supported values in shared util: `{0.5, 1, 2, 2.5, 3, "otsu"}`. The RANBP17 notebooks extended this to arbitrary float K via a regex parser.

**Guidance for choosing K:**
- Higher K → stricter mask, fewer false positives, but risks missing dim signal.
- Lower K → more permissive, better sensitivity for dim markers, but more background.
- Typical starting points: bright/punctate markers (P-bodies, RANBP17) → K ≈ 2–3; diffuse markers (TDP-43 in cytoplasm) → K ≈ 0.5–1.
- Always inspect masks with `plot_masks` before trusting scores.

**KD-condition collapse (important caveat):** in knockdown conditions the target signal may be near noise floor. Adaptive thresholds track the local mean+std, so threshold can be set at noise level → inflated mask → inflated (or unreliable) scores. A fixed-threshold-from-controls variant (compute threshold on control images, apply to KD) was explored at site-level to avoid this. If KD scores look suspiciously high, inspect masks.

---

## 5. Interpretation

| Signal | What it means |
|---|---|
| `fraction_overlap` >> `fraction_overlap_rotated` | Genuine spatial co-localization above chance (judge by **magnitude of Δ**, not just significance) |
| Replicate-level Wilcoxon p < 0.05, alternative='greater' | Real co-localization exceeds rotation null. Note: lenient null → even negative pairs often clear it; rely on positive-vs-negative Δ contrast |
| Effect size (Cohen's d or mixed-effects β) | Magnitude of co-localization difference between groups (KD vs control) |
| CV < 0.1 | Highly reproducible across reps |
| CV > 0.2 | High variability; check for outlier reps or biological heterogeneity |
| Real ≈ rotated | No spatial co-localization beyond what random channel overlap would give |

The Wilcoxon test is the primary result. Effect-size models add interpretability for group comparisons but require ≥2 batches to be reliable.

---

## 6. Adapting to a new experiment — checklist

### Config / constants
- [ ] Set `ANCHOR_MARKER` and `PARTNER_MARKER` names (strings used for df filtering and plot labels).
- [ ] Set `CHANNEL_MAP`: `{marker_name: channel_token}` (e.g. `{'RANBP17': 'ch1', 'TDP43': 'ch3'}`). For site-level this drives filename substitution; for tile-level it drives NPY channel index.
- [ ] Set `MASK_K_MAP`: `{marker_name: K_value}` (float). Tune K per marker per modality by inspecting masks.
- [ ] Set batches, conditions, reps, panels as global filter lists if needed.

### Path parser
- [ ] The shared util path parsers are **positional** and tuned to specific directory depths — they are fragile. Write or rewrite a local parser using **negative indexing** (e.g. `parts[-3]` for condition) so it is robust to prefix depth variation.
- [ ] For tile-level: panel and rep are in the **filename**, not directory. Use regex to extract (e.g. `re.search(r'rep(\d+)', filename)`).
- [ ] For site-level: verify the directory depth matches `sites_paths_to_dataframe`'s 17-component positional unpack, or write a local version.

### Modality choice
- [ ] **Site-level**: apply `rescale_intensity` after `cv2.imread(..., cv2.IMREAD_ANYDEPTH)`. Match anchor/partner by substituting channel token in filename.
- [ ] **Tile-level**: no rescaling. NPY channel 0 = marker, channel 1 = DAPI. Match by (batch, cell_line, condition, rep, site, tile_index). **Confirm both markers are in the same panel** (see same-panel caveat in §3) — cross-panel pairs give 0 shared sites. Verify a shared site has identical DAPI before trusting tile pairing.

### Reuse the generalized tile helpers (preferred for tile-level)
For tile-level work, **import `tools/channels_colocalization/tile_colocalization_utils.py`** instead of copying functions. It is the notebook helpers factored into a module: `list_tile_paths`, `load_tiles_to_dataframe(paths, panels, reps)`, `match_tiles_and_score(df, anchor, partner, mask_k_map, rotation_ks)` (real + rotated, generic markers, arbitrary float K), `show_sample_tiles`, threshold utils. It reuses the shared colormaps. `run_tile_colocalization_controls.py` is a ready example that loops over labeled pairs.

**Import gotcha (NOVA_HOME):** the shared `pixel_colocalization_utils.py` lives under `Collaboration/NOVA`, but the user's shell `NOVA_HOME` may point at `giliwo/NOVA`. `os.environ.setdefault('NOVA_HOME', ...)` then silently keeps the wrong path and the `tools.channels_colocalization.pixel_colocalization_utils` import resolves to the giliwo copy (which lacks it) → `ModuleNotFoundError`. Insert the Collaboration path **explicitly** (`sys.path.insert(1, '/home/projects/hornsteinlab/Collaboration/NOVA')`), independent of `NOVA_HOME`. `tile_colocalization_utils.py` already does this.

### Hard-coding gotchas in shared util (`pixel_colocalization_utils.py`)
- Marker names `DCP1A`, `V5`, `TDP43` and channel tokens `ch2`, `ch3` are hard-coded in `match_tiles_and_score_fraction_overlap` and `match_sites_and_score_fraction_overlap`. For tile-level prefer `tile_colocalization_utils.py` (above); for site-level pass your own marker/channel config or copy and rewrite locally.
- The K enum in `get_threshold` only supports the fixed set `{0.5,1,2,2.5,3,"otsu"}`. The RANBP17 notebooks / `tile_colocalization_utils.py` replaced this with a float-accepting regex version (`mean+Ksd`) — use that for arbitrary K.
- `patient_id` extraction in `load_tiles_to_dataframe` (shared util) uses a cell-line–split heuristic designed for Coyne patient lines; it may not apply to your cell lines. The `tile_colocalization_utils.py` version drops it and parses rep/site/panel from the filename instead.

### Effect-size model
- [ ] Only run `run_analysis_generate_report` if ≥2 batches are present. With 1 batch or singular random effects it falls back to OLS, which is less informative.
- [ ] Set batch as fixed effect if batch effects are expected to be systematic; random effect if batches are random samples from a population.

---

## 7. Example notebooks

| Notebook | Modality | Anchor | Partner | Notes |
|---|---|---|---|---|
| `tools/channels_colocalization/pixel_level_colocalization_TDP43_P-bodies_V5.ipynb` | Site | DCP1A (ch2) | V5/TDP-43 (ch3) | dNLS experiment; 3 batches; K=2.5/2 |
| `tools/channels_colocalization/pixel_level_colocalization_TDP43_P-bodies_Coyne.ipynb` | Tile | DCP1A (K=3) | TDP-43 (K=0.5/1) | Patient cell lines; mixed-effects across patients |
| `tools/channels_colocalization/pixel_level_colocalization_RANBP17_TDP43.ipynb` | Site | RANBP17 (ch1) | TDP-43 (ch3) | RANBP17_exp screen; generalized helpers; negative-index parser |
| `tools/channels_colocalization/pixel_level_colocalization_RANBP17_TDP43_TILES.ipynb` | Tile | RANBP17 | TDP-43 | Same experiment, tile-level; panel/rep from filename; RGB overlay |
| `tools/channels_colocalization/run_tile_colocalization_controls.py` | Tile | (loops) | (loops) | AAT_NOVA_pilot2 positive/negative control validation; loops over labeled within-panel pairs; per-pair CSVs+plots, combined summary, log file |

Shared utilities:
- `/home/projects/hornsteinlab/Collaboration/NOVA/tools/channels_colocalization/pixel_colocalization_utils.py` — original (site-level + colormaps; tile funcs hard-coded to DCP1A/TDP43)
- `<NOVA repo>/tools/channels_colocalization/tile_colocalization_utils.py` — generalized tile-level helpers (preferred for new tile experiments)

---

## Quick-reference formulas

```python
# Threshold
threshold = img.mean() + K * img.std()
mask = img > threshold

# Conditional overlap fraction
score = (anchor_mask & partner_mask).sum() / anchor_mask.sum()
# returns None if anchor_mask.sum() == 0

# Rotation null (average over k=1,2,3 for stability; anchor mask fixed)
score_rotated = np.mean([
    (anchor_mask & (np.rot90(partner_img, k=kk) > threshold_partner)).sum() / anchor_mask.sum()
    for kk in (1, 2, 3)
])

# Wilcoxon (PRIMARY test = replicate level: aggregate tiles to per-(batch,rep) means first)
from scipy.stats import wilcoxon
agg = df_clean.groupby(['batch', 'rep', 'condition'])[['fraction_overlap','fraction_overlap_rotated']].mean()
stat, p = wilcoxon(agg['fraction_overlap'], agg['fraction_overlap_rotated'], alternative='greater')
```

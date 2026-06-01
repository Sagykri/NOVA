# ----------------------------------------------------------------------------- #
# Generalized TILE-level pixel co-localization helpers.
#
# Factored out of pixel_level_colocalization_RANBP17_TDP43_TILES.ipynb so that
# multiple experiments can reuse the same matching/scoring logic instead of
# duplicating it. Works on preprocessed NPY tiles (shape [n_tiles, 100, 100, 2],
# already rescaled to [0,1]; channel 0 = marker, channel 1 = DAPI).
#
# Metric (unchanged from the reference): conditional overlap fraction
#     P(partner+ | anchor+) = |anchor_mask AND partner_mask| / |anchor_mask|
# with a rotation null (partner rotated 90*k deg, averaged over ROTATION_KS).
#
# Per-marker colormaps are reused from the shared pixel_colocalization_utils.
# ----------------------------------------------------------------------------- #
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The shared pixel_colocalization_utils lives in the Collaboration NOVA tree, which
# is NOT necessarily where $NOVA_HOME points. Add it explicitly so the import below
# resolves regardless of the caller's NOVA_HOME (tools is an implicit namespace pkg).
_COLLAB_NOVA = '/home/projects/hornsteinlab/Collaboration/NOVA'
if _COLLAB_NOVA not in sys.path:
    sys.path.insert(1, _COLLAB_NOVA)

# Reuse the shared per-marker colormaps (do not redefine them here).
from tools.channels_colocalization.pixel_colocalization_utils import (
    cmap_black_to_red, cmap_black_to_green, cmap_black_to_blue,
)

# Default rotation angles (in units of 90 deg) for the spatial-random null.
ROTATION_KS = (1, 2, 3)

# Default colormaps; markers without an entry fall back to 'gray' in plots.
DEFAULT_MARKER_CMAP = {
    'anchor':  cmap_black_to_red,
    'partner': cmap_black_to_green,
    'DAPI':    cmap_black_to_blue,
}

_SITE_RE   = re.compile(r'r\d+c\d+f\d+')
_PANEL_RE  = re.compile(r'panel[A-Za-z0-9]+', re.IGNORECASE)
_MASK_K_RE = re.compile(r'^mean\+(-?\d+(?:\.\d+)?)sd$')


# ---------------------------- #
# ----- INPUT DATA UTILS ----- #
# ---------------------------- #

def list_tile_paths(input_dir, batches, cell_lines, conditions, markers):
    """Return NPY tile-file paths under input_dir/<batch>/<cell_line>/<condition>/<marker>/.

    The processed layout has no panel/rep directories (those live in the filename).
    """
    all_paths = []
    for batch in batches:
        for cell_line in cell_lines:
            for cond in conditions:
                for marker in markers:
                    marker_dir = os.path.join(input_dir, batch, cell_line, cond, marker)
                    if not os.path.isdir(marker_dir):
                        continue
                    files = sorted(f for f in os.listdir(marker_dir) if f.endswith('.npy'))
                    all_paths.extend(os.path.join(marker_dir, f) for f in files)
    return all_paths


def _parse_filename(filename):
    """Extract (rep, site, panel) from a processed filename like
    'rep2_r10c14f75-ch3t1_panelA_C9_processed.npy'."""
    rep = filename.split('_', 1)[0] if filename.startswith('rep') else None
    m_site  = _SITE_RE.search(filename)
    m_panel = _PANEL_RE.search(filename)
    return rep, (m_site.group() if m_site else None), (m_panel.group() if m_panel else None)


def load_tiles_to_dataframe(paths, panels=None, reps=None):
    """Explode every NPY (shape [n_tiles, 100, 100, 2]) into ONE ROW PER TILE.

    Metadata: batch/cell_line/condition/marker from the path (negative indexing);
    rep/site/panel from the filename. Optional `panels` / `reps` filter the rows.
    Each row holds the raw tile array in 'tile_data' (no rescaling -- already preprocessed).
    """
    records = []
    for path in paths:
        parts = path.split(os.sep)
        batch, cell_line, condition, marker, filename = (
            parts[-5], parts[-4], parts[-3], parts[-2], parts[-1])
        rep, site, panel = _parse_filename(filename)
        if panels is not None and panel not in panels:
            continue
        if reps is not None and rep not in reps:
            continue
        arr = np.load(path)  # [n_tiles, 100, 100, 2]
        for i in range(arr.shape[0]):
            records.append({
                'path': path, 'batch': batch, 'cell_line': cell_line,
                'panel': panel, 'condition': condition, 'rep': rep,
                'marker': marker, 'filename': filename, 'site': site,
                'tile_index': i, 'tile_data': arr[i],   # (100, 100, 2)
            })
    return pd.DataFrame.from_records(records)


# -------------------------------------- #
# ----- COLOCALIZATION SCORE UTILS ----- #
# -------------------------------------- #

def get_threshold(img, method):
    """Adaptive threshold = mean(img) + K*std(img); K parsed from 'mean+Ksd'."""
    m = _MASK_K_RE.match(method)
    if not m:
        raise ValueError(f"Unsupported threshold method {method!r}. Expected 'mean+Ksd'.")
    k = float(m.group(1))
    return img.mean() + k * img.std()


def method_for_marker(marker, mask_k_map):
    """Return the 'mean+Ksd' method string for `marker` from mask_k_map."""
    if marker not in mask_k_map:
        raise KeyError(f"No MASK_K configured for marker {marker!r}. Add it to mask_k_map.")
    return f"mean+{mask_k_map[marker]}sd"


def score_fraction_overlap(anchor_img, partner_img, anchor_th, partner_th):
    """P(partner+ | anchor+) given numeric thresholds; None if the anchor mask is empty."""
    anchor_mask  = anchor_img  > anchor_th
    partner_mask = partner_img > partner_th
    total_anchor = int(anchor_mask.sum())
    if total_anchor == 0:
        return None
    overlap = int(np.logical_and(anchor_mask, partner_mask).sum())
    return overlap / total_anchor


def match_tiles_and_score(df, anchor_marker, partner_marker, mask_k_map,
                          rotation_ks=ROTATION_KS):
    """For every anchor-marker tile, find the partner-marker tile with the SAME
    (batch, cell_line, condition, rep, site, tile_index) and score `fraction_overlap`
    (real) + `fraction_overlap_rotated` (partner-rotation null averaged over
    rotation_ks). No rescaling. Returns a copy of the anchor tile dataframe with the
    two score columns added.
    """
    anchor_method  = method_for_marker(anchor_marker,  mask_k_map)
    partner_method = method_for_marker(partner_marker, mask_k_map)

    df_anchor  = df[df['marker'] == anchor_marker].copy().reset_index(drop=True)
    df_partner = df[df['marker'] == partner_marker]

    key_cols = ['batch', 'cell_line', 'condition', 'rep', 'site', 'tile_index']
    partner_idx = df_partner.set_index(key_cols)['tile_data'].to_dict()

    fractions, fractions_rot = [], []
    for _, a in df_anchor.iterrows():
        partner_tile = partner_idx.get(tuple(a[c] for c in key_cols))
        if partner_tile is None:
            fractions.append(None); fractions_rot.append(None); continue
        anchor_img  = a['tile_data'][:, :, 0]
        partner_img = partner_tile[:, :, 0]
        anchor_th  = get_threshold(anchor_img,  anchor_method)
        partner_th = get_threshold(partner_img, partner_method)
        fractions.append(score_fraction_overlap(anchor_img, partner_img, anchor_th, partner_th))
        # Rotation null averaged over several 90-deg rotations of the partner.
        # The anchor mask is identical across rotations, so scores are all valid or all None.
        rot_vals = [score_fraction_overlap(anchor_img, np.rot90(partner_img, k=kk),
                                           anchor_th, partner_th) for kk in rotation_ks]
        fractions_rot.append(None if rot_vals[0] is None else float(np.mean(rot_vals)))

    df_anchor['fraction_overlap']         = fractions
    df_anchor['fraction_overlap_rotated'] = fractions_rot
    return df_anchor


# -------------------------- #
# ----- PLOTTING UTILS ----- #
# -------------------------- #

def _pretty(ax):
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _rgb_overlay(anchor_img, partner_img):
    """Signal RGB: anchor -> red, partner -> green. Co-localized signal appears yellow."""
    rgb = np.zeros((*anchor_img.shape, 3), dtype=float)
    rgb[..., 0] = np.clip(anchor_img, 0, 1)
    rgb[..., 1] = np.clip(partner_img, 0, 1)
    return rgb


def _mask_overlay(anchor_mask, partner_mask):
    """Mask RGB: anchor-only red, partner-only green, overlap yellow, background black."""
    rgb = np.zeros((*anchor_mask.shape, 3), dtype=float)
    rgb[..., 0] = anchor_mask.astype(float)
    rgb[..., 1] = partner_mask.astype(float)
    return rgb


def show_sample_tiles(df_overlap, df_all, anchor_marker, partner_marker, mask_k_map,
                      anchor_cmap=None, partner_cmap=None,
                      n_per_condition=2, conditions=None, output_dir=None, random_state=0):
    """Plot per-marker colored tiles + masks + two overlap panels for a few tiles per
    condition. `df_all` is the full per-tile df (to look up partner tiles)."""
    anchor_method  = method_for_marker(anchor_marker,  mask_k_map)
    partner_method = method_for_marker(partner_marker, mask_k_map)
    anchor_cmap  = anchor_cmap  or DEFAULT_MARKER_CMAP['anchor']
    partner_cmap = partner_cmap or DEFAULT_MARKER_CMAP['partner']

    conditions = conditions if conditions is not None else sorted(df_overlap['condition'].unique())
    rng = np.random.default_rng(random_state)

    key_cols = ['batch', 'cell_line', 'condition', 'rep', 'site', 'tile_index']
    partner_idx = df_all[df_all['marker'] == partner_marker].set_index(key_cols)['tile_data'].to_dict()

    for cond in conditions:
        sub = df_overlap[(df_overlap['condition'] == cond) & df_overlap['fraction_overlap'].notna()]
        if sub.empty:
            continue
        sample = sub.sample(n=min(n_per_condition, len(sub)),
                            random_state=int(rng.integers(0, 2**31 - 1)))
        for _, row in sample.iterrows():
            partner_tile = partner_idx.get(tuple(row[c] for c in key_cols))
            if partner_tile is None:
                continue
            anchor_img  = row['tile_data'][:, :, 0]
            partner_img = partner_tile[:, :, 0]
            anchor_th  = get_threshold(anchor_img,  anchor_method)
            partner_th = get_threshold(partner_img, partner_method)
            anchor_mask, partner_mask = anchor_img > anchor_th, partner_img > partner_th

            fig, axs = plt.subplots(1, 6, figsize=(18, 3))
            plt.subplots_adjust(wspace=0.2)
            axs[0].imshow(anchor_img,  cmap=anchor_cmap,  vmin=0, vmax=1); axs[0].set_title(anchor_marker, fontsize=10)
            axs[1].imshow(partner_img, cmap=partner_cmap, vmin=0, vmax=1); axs[1].set_title(partner_marker, fontsize=10)
            axs[2].imshow(anchor_mask,  cmap='gray', vmin=0, vmax=1); axs[2].set_title(f"{anchor_marker} mask (>{anchor_method})", fontsize=9)
            axs[3].imshow(partner_mask, cmap='gray', vmin=0, vmax=1); axs[3].set_title(f"{partner_marker} mask (>{partner_method})", fontsize=9)
            axs[4].imshow(_rgb_overlay(anchor_img, partner_img)); axs[4].set_title("signal overlay\n(R=anchor, G=partner)", fontsize=9)
            axs[5].imshow(_mask_overlay(anchor_mask, partner_mask)); axs[5].set_title("mask overlay\n(overlap=yellow)", fontsize=9)
            for ax in axs:
                _pretty(ax)
            fo, forr = float(row['fraction_overlap']), float(row['fraction_overlap_rotated'])
            plt.suptitle(f"{cond} | {row['rep']} | {row['site']} | tile {row['tile_index']}   |   "
                         f"P({partner_marker}+ | {anchor_marker}+) = {fo:.3f}   (rotated null = {forr:.3f})",
                         fontsize=10, y=1.05)
            plt.tight_layout()
            if output_dir is not None:
                out = os.path.join(output_dir, f"sample_{cond}_{row['rep']}_{row['site']}_t{row['tile_index']}.png")
                plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)

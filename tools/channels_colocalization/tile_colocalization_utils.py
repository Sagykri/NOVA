# ----------------------------------------------------------------------------- #
# Generalized TILE-level pixel co-localization helpers.
#
# Works on preprocessed NPY tiles (shape [n_tiles, 100, 100, 2], already
# rescaled to [0,1]; channel 0 = marker, channel 1 = DAPI).
#
# Metric: conditional overlap fraction
#     P(partner+ | anchor+) = |anchor_mask AND partner_mask| / |anchor_mask|
# with a rotation null (partner rotated 90*k deg, averaged over ROTATION_KS).
# ----------------------------------------------------------------------------- #
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import wilcoxon as _wilcoxon, binomtest

_COLLAB_NOVA = '/home/projects/hornsteinlab/Collaboration/NOVA'
if _COLLAB_NOVA not in sys.path:
    sys.path.insert(1, _COLLAB_NOVA)

from tools.channels_colocalization.pixel_colocalization_utils import (
    cmap_black_to_red, cmap_black_to_green, cmap_black_to_blue,
)

ROTATION_KS = (1, 2, 3)

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
    """Return NPY tile-file paths under input_dir/<batch>/<cell_line>/<condition>/<marker>/."""
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
    print(f"Listed {len(all_paths)} NPY files "
          f"(batches={batches}, conditions={conditions}, markers={markers}).")
    return all_paths


def _parse_filename(filename):
    """Extract (rep, site, panel) from a filename like
    'rep2_r03c04f07-ch1t1_panelA_iW11_processed.npy'."""
    rep = filename.split('_', 1)[0] if filename.startswith('rep') else None
    m_site  = _SITE_RE.search(filename)
    m_panel = _PANEL_RE.search(filename)
    return rep, (m_site.group() if m_site else None), (m_panel.group() if m_panel else None)


def load_tiles_to_dataframe(paths, panels=None, reps=None):
    """Build a per-FILE metadata DataFrame — no tile arrays loaded.

    Each row is one NPY file. Tile arrays are loaded on demand in match_tiles_and_score.
    Optional `panels` / `reps` lists filter rows by filename-parsed values.
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
        records.append({
            'path': path, 'batch': batch, 'cell_line': cell_line,
            'panel': panel, 'condition': condition, 'rep': rep,
            'marker': marker, 'filename': filename, 'site': site,
        })
    df = pd.DataFrame.from_records(records)
    n = len(df)
    print(f"Metadata DataFrame: {n} NPY files. "
          f"Markers: {sorted(df['marker'].unique()) if n else []}. "
          f"panels={panels}, reps={reps}.")
    return df


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
    return int(np.logical_and(anchor_mask, partner_mask).sum()) / total_anchor


def _score_file_pair(arr_a, arr_p, k_anchor, k_partner, rotation_ks):
    """Vectorized scoring of all tiles in a matched file pair.

    Processes the full [n_tiles, 100, 100] batch at once.
    Returns (fractions, fractions_rot) as float32 [n_tiles]; NaN = empty anchor mask.
    """
    a_ch = arr_a[:, :, :, 0].astype(np.float32)
    p_ch = arr_p[:, :, :, 0].astype(np.float32)
    a_th = a_ch.mean(axis=(1, 2)) + k_anchor * a_ch.std(axis=(1, 2))
    p_th = p_ch.mean(axis=(1, 2)) + k_partner * p_ch.std(axis=(1, 2))
    a_mask = a_ch > a_th[:, None, None]
    p_mask = p_ch > p_th[:, None, None]
    anchor_counts = a_mask.sum(axis=(1, 2))
    valid = anchor_counts > 0
    denom = np.where(valid, anchor_counts, 1)
    fractions = np.where(valid, (a_mask & p_mask).sum(axis=(1, 2)) / denom, np.nan)
    rot_stack = []
    for kk in rotation_ks:
        p_rot_mask = np.rot90(p_ch, k=kk, axes=(1, 2)) > p_th[:, None, None]
        rot_stack.append(np.where(valid, (a_mask & p_rot_mask).sum(axis=(1, 2)) / denom, np.nan))
    fractions_rot = np.nanmean(rot_stack, axis=0)
    return fractions.astype(np.float32), fractions_rot.astype(np.float32)


def match_tiles_and_score(df, anchor_marker, partner_marker, mask_k_map,
                          rotation_ks=ROTATION_KS, n_workers=8):
    """Load matched file pairs in parallel, score all tiles per pair with vectorized NumPy.

    Input `df` is the per-FILE metadata DataFrame from load_tiles_to_dataframe (no arrays).
    Returns a per-tile DataFrame with fraction_overlap, fraction_overlap_rotated, partner_path.
    """
    anchor_method  = method_for_marker(anchor_marker,  mask_k_map)
    partner_method = method_for_marker(partner_marker, mask_k_map)
    k_a = float(_MASK_K_RE.match(anchor_method).group(1))
    k_p = float(_MASK_K_RE.match(partner_method).group(1))

    df_anchor  = df[df['marker'] == anchor_marker].reset_index(drop=True)
    df_partner = df[df['marker'] == partner_marker]
    print(f"anchor ({anchor_marker}): {len(df_anchor)} files; "
          f"partner ({partner_marker}): {len(df_partner)} files")
    print(f"  thresholds: anchor='{anchor_method}', partner='{partner_method}'")

    file_key_cols = ['batch', 'cell_line', 'condition', 'rep', 'site']
    meta_cols     = ['path', 'batch', 'cell_line', 'panel', 'condition', 'rep',
                     'marker', 'filename', 'site']
    partner_path_idx = df_partner.set_index(file_key_cols)['path'].to_dict()

    def _process_pair(a_row):
        p_path = partner_path_idx.get(tuple(a_row[c] for c in file_key_cols))
        if p_path is None:
            return None
        arr_a = np.load(a_row['path'])
        arr_p = np.load(p_path)
        if arr_a.shape[0] != arr_p.shape[0]:
            n = min(arr_a.shape[0], arr_p.shape[0])
            arr_a, arr_p = arr_a[:n], arr_p[:n]
        fracs, fracs_rot = _score_file_pair(arr_a, arr_p, k_a, k_p, rotation_ks)
        meta = {c: a_row[c] for c in meta_cols}
        n = arr_a.shape[0]
        return [
            {**meta, 'partner_path': p_path, 'tile_index': i,
             'fraction_overlap':         (None if np.isnan(fracs[i])     else float(fracs[i])),
             'fraction_overlap_rotated': (None if np.isnan(fracs_rot[i]) else float(fracs_rot[i]))}
            for i in range(n)
        ]

    anchor_rows = [row for _, row in df_anchor.iterrows()]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        nested = list(pool.map(_process_pair, anchor_rows))

    records  = [r for sub in nested if sub is not None for r in sub]
    n_miss   = sum(1 for sub in nested if sub is None)
    n_empty  = sum(1 for sub in nested if sub is not None
                   for r in sub if r['fraction_overlap'] is None)
    result = pd.DataFrame.from_records(records)
    print(f"Done. {n_miss} anchor files had no matched partner; "
          f"{n_empty}/{len(result)} tiles had an empty anchor mask.")
    return result


# -------------------------- #
# ----- PLOTTING UTILS ----- #
# -------------------------- #

def _pretty(ax):
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _rgb_overlay(anchor_img, partner_img):
    """Signal RGB: anchor → red, partner → green. Co-localized signal appears yellow."""
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


def show_sample_tiles(df_overlap, anchor_marker, partner_marker, mask_k_map,
                      anchor_cmap=None, partner_cmap=None,
                      n_per_condition=2, conditions=None, output_dir=None, random_state=0):
    """Plot per-marker images + masks + overlays for a few tiles per condition.

    Loads tile arrays on demand from df_overlap['path'] and df_overlap['partner_path'].
    df_overlap must have columns: path, partner_path, tile_index, condition, rep, site,
    fraction_overlap, fraction_overlap_rotated.
    """
    anchor_method  = method_for_marker(anchor_marker,  mask_k_map)
    partner_method = method_for_marker(partner_marker, mask_k_map)
    anchor_cmap  = anchor_cmap  or DEFAULT_MARKER_CMAP['anchor']
    partner_cmap = partner_cmap or DEFAULT_MARKER_CMAP['partner']
    conditions = conditions if conditions is not None else sorted(df_overlap['condition'].unique())
    rng = np.random.default_rng(random_state)

    for cond in conditions:
        sub = df_overlap[(df_overlap['condition'] == cond)
                         & df_overlap['fraction_overlap'].notna()]
        if sub.empty:
            print(f"[{cond}] no scored tiles — skipping.")
            continue
        sample = sub.sample(n=min(n_per_condition, len(sub)),
                            random_state=int(rng.integers(0, 2**31 - 1)))
        for _, row in sample.iterrows():
            i = int(row['tile_index'])
            anchor_img  = np.load(row['path'])[i, :, :, 0]
            partner_img = np.load(row['partner_path'])[i, :, :, 0]
            anchor_th   = get_threshold(anchor_img,  anchor_method)
            partner_th  = get_threshold(partner_img, partner_method)
            anchor_mask  = anchor_img  > anchor_th
            partner_mask = partner_img > partner_th

            fig, axs = plt.subplots(1, 6, figsize=(18, 3))
            plt.subplots_adjust(wspace=0.2)
            axs[0].imshow(anchor_img,  cmap=anchor_cmap,  vmin=0, vmax=1)
            axs[0].set_title(anchor_marker, fontsize=10)
            axs[1].imshow(partner_img, cmap=partner_cmap, vmin=0, vmax=1)
            axs[1].set_title(partner_marker, fontsize=10)
            axs[2].imshow(anchor_mask,  cmap='gray', vmin=0, vmax=1)
            axs[2].set_title(f"{anchor_marker} mask (>{anchor_method})", fontsize=9)
            axs[3].imshow(partner_mask, cmap='gray', vmin=0, vmax=1)
            axs[3].set_title(f"{partner_marker} mask (>{partner_method})", fontsize=9)
            axs[4].imshow(_rgb_overlay(anchor_img, partner_img))
            axs[4].set_title("signal overlay\n(R=anchor, G=partner)", fontsize=9)
            axs[5].imshow(_mask_overlay(anchor_mask, partner_mask))
            axs[5].set_title("mask overlay\n(overlap=yellow)", fontsize=9)
            for ax in axs:
                _pretty(ax)
            fo   = float(row['fraction_overlap'])
            forr = float(row['fraction_overlap_rotated'])
            plt.suptitle(
                f"{cond} | {row['rep']} | {row['site']} | tile {i}   |   "
                f"P({partner_marker}+ | {anchor_marker}+) = {fo:.3f}  "
                f"(rotated null = {forr:.3f})",
                fontsize=10, y=1.05)
            plt.tight_layout()
            if output_dir is not None:
                out = os.path.join(output_dir,
                    f"sample_{cond}_{row['rep']}_{row['site']}_t{i}.png")
                plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig)


def replicate_wilcoxon(df_clean, repl_keys=('batch', 'rep')):
    """Aggregate tiles to per-replicate means and run paired Wilcoxon (real > rotated).

    Returns dict with keys wilcoxon_p / sign_p / n_units / mean_delta,
    compatible with draw_coloc_panel's repl_result parameter.
    Falls back to sign test when n_units < 6 (Wilcoxon p-floor too coarse).
    """
    agg = (df_clean.groupby(list(repl_keys))[['fraction_overlap', 'fraction_overlap_rotated']]
                   .mean().reset_index())
    real   = agg['fraction_overlap'].to_numpy(float)
    rot    = agg['fraction_overlap_rotated'].to_numpy(float)
    diff   = real - rot
    n_eff  = int((diff != 0).sum())
    n_pos  = int((diff > 0).sum())

    out = {
        'n_units':    len(agg),
        'mean_delta': float(diff.mean()) if len(agg) else float('nan'),
        'wilcoxon_p': float('nan'),
        'sign_p':     float('nan'),
    }
    if n_eff >= 2:
        try:
            out['wilcoxon_p'] = float(_wilcoxon(real, rot, alternative='greater').pvalue)
        except ValueError:
            pass
    if n_eff:
        out['sign_p'] = float(binomtest(n_pos, n_eff, 0.5, alternative='greater').pvalue)
    return out


def melt_coloc_df(df):
    """Melt fraction_overlap + fraction_overlap_rotated → long format with 'Real'/'Rotated'."""
    long = df.melt(
        id_vars=['batch', 'condition', 'rep'],
        value_vars=['fraction_overlap', 'fraction_overlap_rotated'],
        var_name='metric', value_name='value',
    )
    long['metric'] = long['metric'].map({
        'fraction_overlap': 'Real',
        'fraction_overlap_rotated': 'Rotated',
    })
    return long


def p_to_stars(p):
    """Convert a p-value float to a significance string."""
    if p is None or not np.isfinite(p):
        return 'descriptive\n(insufficient reps)'
    if p < 0.0001: return '****'
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return f'n.s. (p = {p:.2f})'


def draw_coloc_panel(ax, df_long, color_map, color_by='condition',
                     x_pos_map=None, line_width=1, repl_result=None, show_legend=True):
    """Draw one colocalization panel: boxplot + jittered tile dots + per-rep mean dots.

    Parameters
    ----------
    df_long   : long-format df from melt_coloc_df (columns: batch, condition, rep, metric, value).
    color_map : dict mapping values of `color_by` column to colors.
    color_by  : column used for dot colors and legend labels ('condition' or 'rep').
    x_pos_map : dict mapping 'Real'/'Rotated' to x positions; defaults to {'Real':0,'Rotated':0.3}.
    repl_result : dict from replicate_level_test (keys: wilcoxon_p, sign_p); draws significance
                  bracket when provided.
    show_legend : if True, adds a deduplicated legend to the right of the axis.
    """
    if x_pos_map is None:
        x_pos_map = {'Real': 0, 'Rotated': 0.3}

    for metric, xpos in x_pos_map.items():
        subset = df_long[df_long['metric'] == metric]
        sns.boxplot(
            data=subset, y='value', ax=ax,
            width=0.15, linewidth=line_width,
            showmeans=True, showfliers=False, meanline=True,
            meanprops={'linestyle': '-', 'color': 'black', 'linewidth': line_width},
            boxprops=dict(facecolor='none', edgecolor='black', linewidth=line_width),
            whiskerprops=dict(linewidth=line_width, color='black'),
            capprops=dict(linewidth=line_width, color='black'),
            medianprops=dict(visible=False),
            positions=[xpos],
        )
        x_jit = np.random.normal(loc=xpos, scale=0.08, size=len(subset))
        ax.scatter(x=x_jit, y=subset['value'].values,
                   color='lightgray', s=1, alpha=0.4, zorder=1)

    # Per-(color_by, rep, metric) means as colored dots; deduplicate group cols.
    group_cols = list(dict.fromkeys([color_by, 'rep', 'metric']))
    group_means = df_long.groupby(group_cols)['value'].mean().reset_index()
    for _, row in group_means.iterrows():
        xp = x_pos_map[row['metric']]
        ax.scatter(x=np.random.normal(loc=xp, scale=0.04), y=row['value'],
                   color=color_map.get(row[color_by], 'black'),
                   edgecolor=None, s=12, label=row[color_by], zorder=3)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  title=color_by.capitalize(),
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

    # Significance bracket (optional).
    if repl_result is not None:
        p = repl_result.get('wilcoxon_p', np.nan)
        if not np.isfinite(p):
            p = repl_result.get('sign_p', np.nan)
        ymin, ymax = ax.get_ylim()
        line_y = ymax + 0.01 * ymax
        ax.plot([0, 0, x_pos_map['Rotated'], x_pos_map['Rotated']],
                [line_y, line_y + 0.01 * ymax, line_y + 0.01 * ymax, line_y],
                lw=1.5, c='black')
        ax.text(x_pos_map['Rotated'] / 2, line_y + 0.02 * ymax,
                p_to_stars(p), ha='center', va='bottom', fontsize=8)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - 0.05 * y_range, ymax + 0.1 * y_range)
    ax.set_xlim(min(x_pos_map.values()) - 0.2, max(x_pos_map.values()) + 0.2)
    ax.tick_params(axis='both', which='both', direction='out', length=4, width=1,
                   bottom=True, top=False, left=True, right=False)
    ax.set_xticks(list(x_pos_map.values()))
    ax.set_xticklabels(list(x_pos_map.keys()), rotation=0)
    ax.set_xlabel('')
    sns.despine(ax=ax)

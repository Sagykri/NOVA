#!/usr/bin/env python
"""
Fig. 5 co-localization summary: TDP-43/RANBP17 (untreated) + positive/negative controls.

Three panels (left → right):
  1. TDP-43  → RANBP17   — RANBP17_exp, batch3, untreated, reps 4–8
  2. pDRP1   → TOMM20    — AAT_NOVA_pilot2, batch1–3, C9, Untreated  [positive control]
  3. TOMM20  → pCaMKIIa  — AAT_NOVA_pilot2, batch1–3, C9, Untreated  [negative control]

Each panel: Real vs Rotated-null conditional overlap fraction — open boxplots over
faint per-tile dots, with one colored mean dot per replicate (reps in panel 1,
batches in panels 2–3) and a paired-Wilcoxon (real > rotated) significance bracket.

Per-pair side outputs saved to SAVE_DIR/<anchor>_<partner>/:
  coloc_scores.csv   — per-tile fraction_overlap / fraction_overlap_rotated
  sample_tiles/      — mask overlay figures (n=2 per condition)

Usage:
    conda activate nova
    python fig5_colocalization.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
_GILIWO_NOVA = '/home/projects/hornsteinlab/giliwo/NOVA'
_UTILS_DIR   = os.path.join(_GILIWO_NOVA, 'tools', 'channels_colocalization')
_COLLAB_NOVA = '/home/projects/hornsteinlab/Collaboration/NOVA'
for _p in (_UTILS_DIR, _COLLAB_NOVA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tile_colocalization_utils import (
    list_tile_paths, load_tiles_to_dataframe, match_tiles_and_score,
    melt_coloc_df, show_sample_tiles, replicate_wilcoxon,
)

# ── shared figure style ─────────────────────────────────────────────────────────
sys.path.insert(0, "/home/projects/hornsteinlab/giliwo/RANBP_17_article/figures/figure_2")
from figure_style import apply_style, COLOR, PALETTE, sig_stars, fmt_p  # noqa: E402

# ── Inkscape-editable SVG export helper ─────────────────────────────────────────
sys.path.insert(0, "/home/projects/hornsteinlab/giliwo/.claude/skills/inkscape-converter/scripts")
from inkscape_svg import (  # noqa: E402
    apply_inkscape_rcparams, save_inkscape_svg, validate_editable_svg,
)

# ── save dir ───────────────────────────────────────────────────────────────────
SAVE_DIR = (
    '/home/projects/hornsteinlab/Collaboration/Guy_Lior/RANBP17_exp/'
    'co-localization_and_KD/colocalization_outputs/tiles/fig5'
)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── panel 1 config — RANBP17_exp ───────────────────────────────────────────────
P1_DATA       = os.path.join(_COLLAB_NOVA, 'input', 'images', 'processed', 'RANBP17_exp')
P1_BATCH      = 'batch3'
P1_CELL_LINES = ['iW11']
P1_CONDITION  = 'untreated'
P1_REPS       = ['rep4', 'rep5', 'rep6', 'rep7', 'rep8']
P1_PANELS     = ['panelA']
P1_ANCHOR     = 'TDP-43'
P1_PARTNER    = 'RANBP17'
P1_MASK_K     = {P1_ANCHOR: 1.25, P1_PARTNER: 1.25}

# ── panels 2 & 3 config — AAT_NOVA_pilot2 ─────────────────────────────────────
AAT_DATA       = os.path.join(_COLLAB_NOVA, 'input', 'images', 'processed',
                               'AAT_NOVA_pilot2', 'processed')
AAT_BATCHES    = ['batch1', 'batch2', 'batch3']
AAT_CELL_LINES = ['C9']
AAT_CONDITION  = 'Untreated'

P2_ANCHOR  = 'pDRP1';   P2_PARTNER  = 'TOMM20'
P3_ANCHOR  = 'TOMM20';  P3_PARTNER  = 'pCaMKIIa'
AAT_MASK_K = {'pDRP1': 1.5, 'TOMM20': 1.5, 'pCaMKIIa': 1.5}

# Real/Rotated x positions and jitter widths
_XPOS   = {'Real': 0.0, 'Rotated': 0.6}
_DOT_JIT = 0.05   # per-rep mean dots
_TILE_JIT = 0.09  # faint per-tile dots


# ── load / score / save helper ─────────────────────────────────────────────────
def load_score_save(data_path, batches, cell_lines, conditions, panels, reps,
                    anchor, partner, mask_k_map):
    """Load, score, and persist side outputs for one marker pair; returns df_clean."""
    pair_dir   = os.path.join(SAVE_DIR, f'{anchor}_{partner}')
    tiles_dir  = os.path.join(pair_dir, 'sample_tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    paths = list_tile_paths(data_path, batches, cell_lines, conditions, [anchor, partner])
    df    = load_tiles_to_dataframe(paths, panels=panels, reps=reps)

    df_overlap = match_tiles_and_score(df, anchor, partner, mask_k_map)
    df_clean   = df_overlap.dropna(
        subset=['fraction_overlap', 'fraction_overlap_rotated']
    ).copy()
    df_clean['fraction_overlap']         = df_clean['fraction_overlap'].astype(float)
    df_clean['fraction_overlap_rotated'] = df_clean['fraction_overlap_rotated'].astype(float)
    print(f"  scored tiles after dropna: {len(df_clean)}")

    csv_path = os.path.join(pair_dir, 'coloc_scores.csv')
    df_overlap.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    show_sample_tiles(
        df_overlap, anchor, partner, mask_k_map,
        n_per_condition=2, output_dir=tiles_dir,
    )
    return df_clean


def prepare_data():
    """Score all three marker pairs; return (longs, stats, color specs) per panel."""
    print("=== Panel 1: TDP-43 → RANBP17 (RANBP17_exp, batch3, untreated) ===")
    df1 = load_score_save(P1_DATA, [P1_BATCH], P1_CELL_LINES, [P1_CONDITION],
                          P1_PANELS, P1_REPS, P1_ANCHOR, P1_PARTNER, P1_MASK_K)
    print("\n=== Panel 2: pDRP1 → TOMM20 [+ctrl] (AAT_NOVA_pilot2, Untreated) ===")
    df2 = load_score_save(AAT_DATA, AAT_BATCHES, AAT_CELL_LINES, [AAT_CONDITION],
                          None, None, P2_ANCHOR, P2_PARTNER, AAT_MASK_K)
    print("\n=== Panel 3: TOMM20 → pCaMKIIa [-ctrl] (AAT_NOVA_pilot2, Untreated) ===")
    df3 = load_score_save(AAT_DATA, AAT_BATCHES, AAT_CELL_LINES, [AAT_CONDITION],
                          None, None, P3_ANCHOR, P3_PARTNER, AAT_MASK_K)

    stats1 = replicate_wilcoxon(df1, repl_keys=('batch', 'rep'))
    stats2 = replicate_wilcoxon(df2, repl_keys=('batch', 'rep'))
    stats3 = replicate_wilcoxon(df3, repl_keys=('batch', 'rep'))

    # color-key per panel: panel 1 by rep, panels 2-3 by batch
    panels = [
        dict(long=melt_coloc_df(df1), color_by='rep',
             title=f'{P1_ANCHOR} → {P1_PARTNER}\n(untreated)',          stats=stats1),
        dict(long=melt_coloc_df(df2), color_by='batch',
             title=f'{P2_ANCHOR} → {P2_PARTNER}\n[positive ctrl]',      stats=stats2),
        dict(long=melt_coloc_df(df3), color_by='batch',
             title=f'{P3_ANCHOR} → {P3_PARTNER}\n[negative ctrl]',      stats=stats3),
    ]
    return panels


# ── single-panel drawer (figure_style compliant) ────────────────────────────────
def _category_colors(values):
    """Map a sorted list of category labels to the colorblind-safe PALETTE."""
    vals = sorted(values)
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(vals)}


def _draw_panel(ax, df_long, color_by, title, stats, show_legend=False):
    """Real vs Rotated overlap: open boxplots, faint tile dots, per-rep mean dots."""
    cmap_cat = _category_colors(df_long[color_by].unique())

    box_kw = dict(
        widths=0.18, showmeans=True, showfliers=False, meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor='none', edgecolor=COLOR['fit'], linewidth=1.0),
        whiskerprops=dict(color=COLOR['fit'], linewidth=1.0),
        capprops=dict(color=COLOR['fit'], linewidth=1.0),
        medianprops=dict(visible=False),
        meanprops=dict(linestyle='-', color=COLOR['fit'], linewidth=1.4),
    )
    for metric, xpos in _XPOS.items():
        vals = df_long.loc[df_long['metric'] == metric, 'value'].to_numpy(float)
        if vals.size == 0:
            continue
        ax.boxplot([vals], positions=[xpos], **box_kw)
        x_jit = np.random.normal(loc=xpos, scale=_TILE_JIT, size=vals.size)
        ax.scatter(x_jit, vals, color=COLOR['light'], s=2, alpha=0.45,
                   linewidths=0, zorder=1)

    # one colored mean dot per (category, metric)
    means = df_long.groupby([color_by, 'metric'])['value'].mean().reset_index()
    for _, row in means.iterrows():
        xp = _XPOS[row['metric']]
        ax.scatter(np.random.normal(loc=xp, scale=_DOT_JIT), row['value'],
                   color=cmap_cat[row[color_by]], edgecolor='white',
                   linewidths=0.4, s=26, zorder=3, label=row[color_by])

    # significance bracket in the top margin: paired Wilcoxon (real > rotated),
    # sign-test fallback. Placed above the data range (set by the caller's ylim).
    p = stats.get('wilcoxon_p', np.nan)
    if not np.isfinite(p):
        p = stats.get('sign_p', np.nan)
    stars = sig_stars(p)
    label = f"{stars} {fmt_p(p)}".strip() if stars else f"n.s. ({fmt_p(p)})"
    y0, x0, x1 = 1.0, _XPOS['Real'], _XPOS['Rotated']
    ax.plot([x0, x0, x1, x1], [y0, y0 + 0.025, y0 + 0.025, y0],
            lw=1.0, c=COLOR['fit'], clip_on=False)
    ax.text((x0 + x1) / 2, y0 + 0.045, label, ha='center', va='bottom',
            fontsize=8, color=COLOR['fit'], clip_on=False)

    ax.set_title(title)
    ax.set_xlim(_XPOS['Real'] - 0.35, _XPOS['Rotated'] + 0.35)
    ax.set_xticks(list(_XPOS.values()))
    ax.set_xticklabels(list(_XPOS.keys()))
    ax.set_xlabel('')

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  title=color_by.capitalize(), loc='upper right',
                  fontsize=7.5, title_fontsize=8, handletextpad=0.2)


# ── plot_panel / main ───────────────────────────────────────────────────────────
def plot_panel(fig=None, axes=None, return_stats=False, panels=None):
    """Render the 3-panel co-localization summary.

    axes : optional length-3 sequence of Axes to draw into (for composing into a
           larger paper figure). When None, a standalone 1×3 figure is created.
    panels : optional pre-scored panel specs from prepare_data(); scored on demand
             if omitted.
    """
    apply_style()
    if panels is None:
        panels = prepare_data()

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
        fig.subplots_adjust(wspace=0.08)
    elif fig is None:
        fig = axes[0].get_figure()

    for i, (ax, spec) in enumerate(zip(axes, panels)):
        _draw_panel(ax, spec['long'], spec['color_by'], spec['title'],
                    spec['stats'], show_legend=(i == 0))

    axes[0].set_ylabel('Colocalization level\n(fraction overlap)')
    for ax in axes[1:]:
        ax.set_ylabel('')
    axes[0].set_ylim(-0.05, 1.18)   # headroom for the significance brackets

    stats_df = pd.DataFrame([
        dict(pair=spec['title'].splitlines()[0], **spec['stats'])
        for spec in panels
    ])

    if own_fig:
        fig.tight_layout()
    if return_stats:
        return fig, stats_df
    return fig


def main():
    fig, stats_df = plot_panel(return_stats=True)
    stem = 'fig5_colocalization'
    for ext in ('png', 'pdf'):
        out = os.path.join(SAVE_DIR, f'{stem}.{ext}')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"[ok] wrote {out}")

    # Inkscape-editable SVG: set rcParams AFTER plot_panel()'s apply_style(), else
    # the style reset clobbers svg.fonttype='none' and text outlines to paths.
    apply_inkscape_rcparams()
    svg = save_inkscape_svg(fig, os.path.join(SAVE_DIR, f'{stem}.svg'))
    validate_editable_svg(svg)
    print(f"[ok] wrote {svg}")
    plt.close(fig)

    tsv = os.path.join(SAVE_DIR, f'{stem}_stats.tsv')
    stats_df.to_csv(tsv, sep='\t', index=False)
    print(f"[ok] wrote {tsv}")


if __name__ == '__main__':
    main()

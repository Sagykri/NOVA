"""
Fig. 5 co-localization summary: TDP-43/RANBP17 (untreated) + positive/negative controls.

Three panels (left → right):
  1. TDP-43  → RANBP17   — RANBP17_exp, batch3, untreated, reps 4–8
  2. pDRP1   → TOMM20    — AAT_NOVA_pilot2, batch1–3, C9, Untreated  [positive control]
  3. TOMM20  → pCaMKIIa  — AAT_NOVA_pilot2, batch1–3, C9, Untreated  [negative control]

Per-pair outputs saved to SAVE_DIR/<anchor>_<partner>/:
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
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────────
_GILIWO_NOVA = '/home/projects/hornsteinlab/giliwo/NOVA'
_UTILS_DIR   = os.path.join(_GILIWO_NOVA, 'tools', 'channels_colocalization')
_COLLAB_NOVA = '/home/projects/hornsteinlab/Collaboration/NOVA'
for _p in (_UTILS_DIR, _COLLAB_NOVA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tile_colocalization_utils import (
    list_tile_paths, load_tiles_to_dataframe, match_tiles_and_score,
    melt_coloc_df, draw_coloc_panel, show_sample_tiles, replicate_wilcoxon,
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

# ── font ───────────────────────────────────────────────────────────────────────
_FONT = '/home/projects/hornsteinlab/sagyk/anaconda3/envs/nova/fonts/arial.ttf'
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 8})


# ── load / score / save helper ─────────────────────────────────────────────────
def load_score_save(data_path, batches, cell_lines, conditions, panels, reps,
                    anchor, partner, mask_k_map):
    """Load, score, and persist outputs for one marker pair; returns df_clean."""
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


# ── load & score ───────────────────────────────────────────────────────────────
print("=== Panel 1: TDP-43 → RANBP17 (RANBP17_exp, batch3, untreated) ===")
df1 = load_score_save(
    P1_DATA, [P1_BATCH], P1_CELL_LINES, [P1_CONDITION],
    P1_PANELS, P1_REPS, P1_ANCHOR, P1_PARTNER, P1_MASK_K,
)

print("\n=== Panel 2: pDRP1 → TOMM20 [+ctrl] (AAT_NOVA_pilot2, Untreated) ===")
df2 = load_score_save(
    AAT_DATA, AAT_BATCHES, AAT_CELL_LINES, [AAT_CONDITION],
    None, None, P2_ANCHOR, P2_PARTNER, AAT_MASK_K,
)

print("\n=== Panel 3: TOMM20 → pCaMKIIa [-ctrl] (AAT_NOVA_pilot2, Untreated) ===")
df3 = load_score_save(
    AAT_DATA, AAT_BATCHES, AAT_CELL_LINES, [AAT_CONDITION],
    None, None, P3_ANCHOR, P3_PARTNER, AAT_MASK_K,
)

# ── replicate-level statistics (real > rotated, paired Wilcoxon) ──────────────
stats1 = replicate_wilcoxon(df1, repl_keys=('batch', 'rep'))
stats2 = replicate_wilcoxon(df2, repl_keys=('batch', 'rep'))
stats3 = replicate_wilcoxon(df3, repl_keys=('batch', 'rep'))
for label, s in [('TDP-43→RANBP17', stats1), ('pDRP1→TOMM20', stats2), ('TOMM20→pCaMKIIa', stats3)]:
    print(f"  {label}: n={s['n_units']} units, Δ={s['mean_delta']:+.4f}, "
          f"Wilcoxon p={s['wilcoxon_p']:.3g}, sign p={s['sign_p']:.3g}")

# ── long format + color maps ───────────────────────────────────────────────────
long1 = melt_coloc_df(df1)
long2 = melt_coloc_df(df2)
long3 = melt_coloc_df(df3)

# Panel 1: color by rep (reps 4–8 within batch3)
rep_order  = sorted(df1['rep'].unique())
cmap_rep   = dict(zip(rep_order, sns.color_palette('Set2', n_colors=len(rep_order))))

# Panels 2 & 3: color by batch (batch1–3 across three batches)
batch_order = sorted(df2['batch'].unique())
cmap_batch  = dict(zip(batch_order, sns.color_palette('Set2', n_colors=len(batch_order))))

# ── figure ─────────────────────────────────────────────────────────────────────
sns.set_style('white')
fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
fig.subplots_adjust(wspace=0.08)

PANELS_CFG = [
    (axes[0], long1, cmap_rep,   'rep',   f'{P1_ANCHOR} → {P1_PARTNER}\n(untreated)',        stats1, False),
    (axes[1], long2, cmap_batch, 'batch', f'{P2_ANCHOR} → {P2_PARTNER}\n[positive ctrl]',    stats2, False),
    (axes[2], long3, cmap_batch, 'batch', f'{P3_ANCHOR} → {P3_PARTNER}\n[negative ctrl]',    stats3, True),
]
for ax, df_long, color_map, color_by, title, stats, show_leg in PANELS_CFG:
    draw_coloc_panel(ax, df_long, color_map, color_by=color_by, repl_result=stats, show_legend=show_leg)
    ax.set_title(title, fontsize=9)

axes[0].set_ylabel('Colocalization level\n(fraction overlap)')
for ax in axes[1:]:
    ax.set_ylabel('')
axes[0].set_ylim(-0.05, 1)

plt.tight_layout()

stem = 'fig5_colocalization'
for ext in ('eps', 'png'):
    out = os.path.join(SAVE_DIR, f'{stem}.{ext}')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")
plt.show()

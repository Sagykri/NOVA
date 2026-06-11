"""
Publication-quality TDP-43 / RANBP17 colocalization plot, faceted per condition.

Layout: 3 panels (control-179 | control-180 | untreated), each showing Real vs Rotated
side-by-side.  Grey dots = individual tiles; coloured dots = per-rep mean (one dot per rep,
reps 4-8); box/whisker per the step-5 notebook style.  Shared y-axis across panels.

Usage:
    conda activate nova
    python plot_coloc_per_condition_batch3.py
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
_GILIWO_NOVA  = '/home/projects/hornsteinlab/giliwo/NOVA'
_UTILS_DIR    = os.path.join(_GILIWO_NOVA, 'tools', 'channels_colocalization')
_COLLAB_NOVA  = '/home/projects/hornsteinlab/Collaboration/NOVA'
for _p in (_UTILS_DIR, _COLLAB_NOVA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tile_colocalization_utils import (
    list_tile_paths, load_tiles_to_dataframe, match_tiles_and_score,
    melt_coloc_df, draw_coloc_panel,
)

# ── experiment constants ────────────────────────────────────────────────────────
MAIN_DATA_PATH = os.path.join(_COLLAB_NOVA, 'input', 'images', 'processed', 'RANBP17_exp')

BATCH      = 'batch3'
CELL_LINES = ['iW11']
REPS       = ['rep4', 'rep5', 'rep6', 'rep7', 'rep8']
PANELS     = ['panelA']
CONDITIONS = ['control-179', 'control-180', 'untreated']   # panel order (left → right)

ANCHOR_MARKER  = 'TDP-43'
PARTNER_MARKER = 'RANBP17'
MASK_K_MAP     = {ANCHOR_MARKER: 1.25, PARTNER_MARKER: 1.25}

SAVE_DIR = (
    f'/home/projects/hornsteinlab/Collaboration/Guy_Lior/RANBP17_exp/'
    f'co-localization_and_KD/colocalization_outputs/tiles/{BATCH}/'
    f'{ANCHOR_MARKER}_{PARTNER_MARKER}/'
    f'ancK{MASK_K_MAP[ANCHOR_MARKER]}_partK{MASK_K_MAP[PARTNER_MARKER]}'
)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── font ───────────────────────────────────────────────────────────────────────
_FONT = '/home/projects/hornsteinlab/sagyk/anaconda3/envs/nova/fonts/arial.ttf'
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 8})

# ── load & score ───────────────────────────────────────────────────────────────
print("Listing paths …")
paths = list_tile_paths(
    MAIN_DATA_PATH,
    batches=[BATCH],
    cell_lines=CELL_LINES,
    conditions=CONDITIONS,
    markers=[ANCHOR_MARKER, PARTNER_MARKER],
)
df = load_tiles_to_dataframe(paths, panels=PANELS, reps=REPS)

print("\nScoring …")
df_overlap = match_tiles_and_score(df, ANCHOR_MARKER, PARTNER_MARKER, MASK_K_MAP)

df_clean = df_overlap.dropna(subset=['fraction_overlap', 'fraction_overlap_rotated']).copy()
df_clean['fraction_overlap']         = df_clean['fraction_overlap'].astype(float)
df_clean['fraction_overlap_rotated'] = df_clean['fraction_overlap_rotated'].astype(float)
df_clean = df_clean[df_clean['condition'].isin(CONDITIONS)]
print(f"\nScored tiles after dropna: {len(df_clean)}")

# ── build long-format df & color map ──────────────────────────────────────────
df_long = melt_coloc_df(df_clean)

# Color by rep (5 reps, each gets a distinct hue within each condition panel).
rep_order = sorted(df_clean['rep'].unique())
color_map = dict(zip(rep_order, sns.color_palette('Set2', n_colors=len(rep_order))))

# ── plot ───────────────────────────────────────────────────────────────────────
sns.set_style('white')
fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
fig.subplots_adjust(wspace=0.08)

for ax, cond in zip(axes, CONDITIONS):
    subset = df_long[df_long['condition'] == cond]
    draw_coloc_panel(
        ax, subset, color_map,
        color_by='rep',
        show_legend=(ax is axes[-1]),   # legend only on the rightmost panel
    )
    ax.set_title(cond, fontsize=9)

axes[0].set_ylabel('Colocalization level\n(fraction overlap)')
for ax in axes[1:]:
    ax.set_ylabel('')

fig.suptitle(f'{ANCHOR_MARKER} – {PARTNER_MARKER} Colocalization · {BATCH} · reps 4–8',
             fontsize=9, y=1.02)
axes[0].set_ylim(-0.05, 1)   # shared via sharey=True

plt.tight_layout()
stem = f'coloc_{ANCHOR_MARKER}_{PARTNER_MARKER}_per_condition_{BATCH}'
for ext in ('eps', 'png'):
    out = os.path.join(SAVE_DIR, f'{stem}.{ext}')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")
plt.show()

#!/usr/bin/env python
# ----------------------------------------------------------------------------- #
# Tile-level pixel co-localization -- positive / negative control validation.
#
# Loops over a list of (anchor, partner) marker pairs, each labeled POSITIVE or
# NEGATIVE control, and runs the SAME tile-level method as
# pixel_level_colocalization_RANBP17_TDP43_TILES.ipynb:
#   - per-tile conditional overlap fraction  P(partner+ | anchor+)
#   - rotation null (partner rotated 90*k deg, averaged over ROTATION_KS)
#   - replicate-level paired Wilcoxon (real > rotated), the primary test
#   - per-tile descriptive Wilcoxon + coefficient of variation per condition
#
# Dataset: AAT_NOVA_pilot2, C9 cell line, non-targeting (NT) conditions only.
#
# IMPORTANT (tile-level validity): two markers can only be paired per-tile when
# they are different channels of the SAME panel (shared nucleus segmentation /
# DAPI). All pairs below are within-panel; cross-panel pairs are not feasible.
#
# Outputs (under OUTPUT_ROOT):
#   <positive|negative>/<ANCHOR>_<PARTNER>/<thresholds>/   per-pair CSVs + plots
#   combined_summary.csv                                   one row per pair
#   run_<timestamp>.log                                    run log
#
# Usage:
#   conda activate nova
#   python run_tile_colocalization_controls.py
# ----------------------------------------------------------------------------- #
import os
import sys
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import wilcoxon, binomtest

# Local generalized tile helpers live next to this script; the helpers module adds
# the Collaboration NOVA tree (for the shared colormaps) to sys.path on its own.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tile_colocalization_utils import (
    list_tile_paths, load_tiles_to_dataframe, match_tiles_and_score,
    show_sample_tiles, ROTATION_KS,
)

# ----------------------------------------------------------------------------- #
# ----- CONFIG ----- #
# ----------------------------------------------------------------------------- #
MAIN_DATA_PATH = ("/home/projects/hornsteinlab/Collaboration/NOVA/"
                  "input/images/processed/AAT_NOVA_pilot2/processed")
OUTPUT_ROOT = ("/home/projects/hornsteinlab/giliwo/NOVA/tools/channels_colocalization/colocalization_outputs/"
               "AAT_NOVA_pilot2_tiles_controls")

BATCHES    = ['batch1', 'batch2', 'batch3']
CELL_LINES = ['C9']
CONDITIONS = ['NT-1873', 'NT-6301-3085']   # non-targeting only
PANELS = None   # markers are panel-specific; no extra filter needed
REPS   = None   # use all reps

# Adaptive threshold K (mean + K*sd) per marker. Default 1.5 == reference method.
MASK_K_MAP = {
    'pDRP1':   1.5,
    'TOMM20':  1.5,
    'pCaMKIIa': 1.5,
    'TDP-43':  1.5,
    'SMI32':   1.5,
    'pTDP-43': 1.5,
    'ATF4':    1.5,
}

# Pairs to validate. anchor = denominator (the more spatially-restricted marker).
# Every pair is within a single panel so tile_index pairing is valid.
PAIRS = [
    # POSITIVE control
    {'control': 'positive', 'panel': 'panelB', 'anchor': 'pDRP1',   'partner': 'TOMM20',
     'rationale': 'Both mitochondrial; pDRP1 marks fission sites on the mito network.'},
    # NEGATIVE controls -- one per panel
    {'control': 'negative', 'panel': 'panelA', 'anchor': 'TDP-43',  'partner': 'SMI32',
     'rationale': 'Nuclear RNA-binding protein vs axonal neurofilament.'},
    {'control': 'negative', 'panel': 'panelB', 'anchor': 'TOMM20',  'partner': 'pCaMKIIa',
     'rationale': 'Mitochondria vs synaptic kinase (within-panel contrast to the positive).'},
    {'control': 'negative', 'panel': 'panelC', 'anchor': 'pTDP-43', 'partner': 'ATF4',
     'rationale': 'Cytoplasmic phospho-TDP-43 vs nuclear transcription factor ATF4.'},
]

REPL_KEYS = ['batch', 'rep', 'condition']   # one replicate unit per (batch, rep, condition)


# ----------------------------------------------------------------------------- #
# ----- HELPERS ----- #
# ----------------------------------------------------------------------------- #
def setup_logging(output_root):
    os.makedirs(output_root, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_root, f'run_{ts}.log')
    logger = logging.getLogger('tile_coloc')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger, log_path


def thresholds_tag(anchor, partner):
    return f"ancK{MASK_K_MAP[anchor]}_partK{MASK_K_MAP[partner]}"


def replicate_level_test(df_clean):
    """Aggregate tiles to per-(batch, rep, condition) means and test real > rotated.
    Replicate is the unit (tiles are pseudoreplicates). Returns a dict of stats."""
    agg = (df_clean.groupby(REPL_KEYS)[['fraction_overlap', 'fraction_overlap_rotated']]
                   .mean().reset_index())
    real = agg['fraction_overlap'].to_numpy(float)
    rot  = agg['fraction_overlap_rotated'].to_numpy(float)
    diff = real - rot
    n_units = len(agg)
    n_pos, n_eff = int((diff > 0).sum()), int((diff != 0).sum())
    out = {
        'n_repl_units': n_units, 'n_batches': df_clean['batch'].nunique(),
        'repl_mean_real': float(real.mean()) if n_units else float('nan'),
        'repl_mean_rotated': float(rot.mean()) if n_units else float('nan'),
        'repl_mean_delta': float(diff.mean()) if n_units else float('nan'),
        'repl_sign_p': float('nan'), 'repl_wilcoxon_p': float('nan'),
    }
    if n_eff:
        out['repl_sign_p'] = float(binomtest(n_pos, n_eff, 0.5, alternative='greater').pvalue)
    try:
        out['repl_wilcoxon_p'] = float(wilcoxon(real, rot, alternative='greater').pvalue)
    except ValueError:
        pass  # all-zero differences etc.
    return out, agg


def tile_level_wilcoxon(df_clean):
    """Descriptive per-tile paired Wilcoxon (real > rotated). Pseudoreplicated."""
    try:
        return float(wilcoxon(df_clean['fraction_overlap'].to_numpy(float),
                              df_clean['fraction_overlap_rotated'].to_numpy(float),
                              alternative='greater').pvalue)
    except ValueError:
        return float('nan')


def cv_per_condition(df_clean):
    """Coefficient of variation of real fraction_overlap across replicates, per condition.
    Returns (worst CV across conditions, dict per condition)."""
    per_rep = (df_clean.groupby(['condition', 'batch', 'rep'])['fraction_overlap']
                       .mean().reset_index())
    stats = (per_rep.groupby('condition')['fraction_overlap']
                    .agg(['mean', 'std'])
                    .assign(CV=lambda d: d['std'] / d['mean']))
    cv_map = stats['CV'].to_dict()
    worst = float(np.nanmax(list(cv_map.values()))) if cv_map else float('nan')
    return worst, cv_map


def plot_pair(df_clean, agg, pair, save_path):
    """Box+jitter of real vs rotated (per-tile pool, per-rep means overlaid) for one pair."""
    anchor, partner = pair['anchor'], pair['partner']
    long = df_clean.melt(id_vars=['batch', 'condition', 'rep'],
                         value_vars=['fraction_overlap', 'fraction_overlap_rotated'],
                         var_name='metric', value_name='value')
    long['metric'] = long['metric'].map({'fraction_overlap': 'Real',
                                         'fraction_overlap_rotated': 'Rotated'})
    fig, ax = plt.subplots(figsize=(3.2, 4))
    x_pos = {'Real': 0, 'Rotated': 0.3}
    for metric, xp in x_pos.items():
        d = long[long['metric'] == metric]
        sns.boxplot(data=d, y='value', ax=ax, width=0.15, linewidth=1,
                    showmeans=True, showfliers=False, meanline=True,
                    meanprops={'linestyle': '-', 'color': 'black', 'linewidth': 1},
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1),
                    whiskerprops=dict(linewidth=1, color='black'),
                    capprops=dict(linewidth=1, color='black'),
                    medianprops=dict(visible=False), positions=[xp])
        xj = np.random.normal(xp, 0.08, size=len(d))
        ax.scatter(xj, d['value'], color='lightgray', s=1, alpha=0.4, zorder=1)
    # per-(condition,rep) means as colored dots
    gmeans = long.groupby(['condition', 'rep', 'metric'])['value'].mean().reset_index()
    conds = sorted(gmeans['condition'].unique())
    cmap = dict(zip(conds, sns.color_palette('Set2', n_colors=len(conds))))
    for _, r in gmeans.iterrows():
        ax.scatter(np.random.normal(x_pos[r['metric']], 0.04), r['value'],
                   color=cmap.get(r['condition'], 'black'), s=12, zorder=3, label=r['condition'])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Condition',
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.set_xticks(list(x_pos.values())); ax.set_xticklabels(list(x_pos.keys()))
    ax.set_ylim(-0.05, 1); ax.set_ylabel('Colocalization (fraction overlap)'); ax.set_xlabel('')
    ax.set_title(f"[{pair['control']}] {anchor}→{partner} ({pair['panel']})", fontsize=9)
    sns.despine(); plt.tight_layout()
    out = os.path.join(save_path, f'coloc_{anchor}_{partner}_tiles.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    return out


def plot_combined(summary, output_root):
    """Bar chart of mean real vs rotated per pair, grouped by control type."""
    s = summary.sort_values(['control', 'tile_mean_real'], ascending=[True, False]).reset_index(drop=True)
    labels = [f"{r.anchor}→{r.partner}\n({r.control[:3]}, {r.panel})" for r in s.itertuples()]
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(s)), 4))
    colors = ['#2e7d32' if c == 'positive' else '#c0392b' for c in s['control']]
    ax.bar(x - w/2, s['tile_mean_real'], w, color=colors, edgecolor='black', label='Real')
    ax.bar(x + w/2, s['tile_mean_rotated'], w, color=colors, edgecolor='black',
           hatch='//', alpha=0.55, label='Rotated null')
    for xi, p in zip(x, s['repl_wilcoxon_p']):
        star = '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'
        ax.text(xi, max(s.loc[xi, 'tile_mean_real'], s.loc[xi, 'tile_mean_rotated']) + 0.02,
                star, ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Mean fraction overlap'); ax.set_ylim(0, 1)
    ax.set_title('Tile-level co-localization: real vs rotated null (replicate Wilcoxon stars)')
    handles = [mpatches.Patch(facecolor='gray', edgecolor='black', label='Real'),
               mpatches.Patch(facecolor='gray', edgecolor='black', hatch='//', alpha=0.55, label='Rotated null'),
               mpatches.Patch(facecolor='#2e7d32', edgecolor='black', label='positive'),
               mpatches.Patch(facecolor='#c0392b', edgecolor='black', label='negative')]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    sns.despine(); plt.tight_layout()
    out = os.path.join(output_root, 'combined_summary_barplot.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- #
# ----- MAIN ----- #
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='Tile-level coloc control validation.')
    ap.add_argument('--no-sample-tiles', action='store_true',
                    help='Skip the per-pair sample-tile mask figures (faster).')
    args = ap.parse_args()

    logger, log_path = setup_logging(OUTPUT_ROOT)
    logger.info('=== Tile-level co-localization control validation ===')
    logger.info(f'Data: {MAIN_DATA_PATH}')
    logger.info(f'Scope: cell_lines={CELL_LINES}, conditions={CONDITIONS}, batches={BATCHES}')
    logger.info(f'Rotation null Ks: {ROTATION_KS}; replicate unit = {REPL_KEYS}')
    logger.info(f'Output root: {OUTPUT_ROOT}')
    logger.info(f'{len(PAIRS)} pairs: ' +
                ', '.join(f"[{p['control']}] {p['anchor']}->{p['partner']}" for p in PAIRS))

    summary_rows = []
    for pair in PAIRS:
        anchor, partner, control = pair['anchor'], pair['partner'], pair['control']
        tag = thresholds_tag(anchor, partner)
        save_path = os.path.join(OUTPUT_ROOT, control, f'{anchor}_{partner}', tag)
        os.makedirs(save_path, exist_ok=True)
        logger.info('-' * 70)
        logger.info(f"PAIR [{control}] {anchor} (anchor) -> {partner} (partner) | "
                    f"{pair['panel']} | K={tag}")
        logger.info(f"  rationale: {pair['rationale']}")

        # --- Load only the two markers we need for this pair.
        paths = list_tile_paths(MAIN_DATA_PATH, BATCHES, CELL_LINES, CONDITIONS, [anchor, partner])
        df = load_tiles_to_dataframe(paths, panels=PANELS, reps=REPS)
        if df.empty:
            logger.warning(f"  no tiles found for {anchor}/{partner}; skipping.")
            continue
        counts = df.groupby('marker').size().to_dict()
        logger.info(f"  tiles loaded: {counts}")

        # --- Score real + rotated overlap per tile.
        df_overlap = match_tiles_and_score(df, anchor, partner, MASK_K_MAP, rotation_ks=ROTATION_KS)
        df_clean = df_overlap.dropna(subset=['fraction_overlap', 'fraction_overlap_rotated']).copy()
        df_clean['fraction_overlap'] = df_clean['fraction_overlap'].astype(float)
        df_clean['fraction_overlap_rotated'] = df_clean['fraction_overlap_rotated'].astype(float)
        n_scored, n_total = len(df_clean), len(df_overlap)
        logger.info(f"  scored tiles: {n_scored}/{n_total} (rest had no partner / empty anchor mask)")
        if n_scored == 0:
            logger.warning("  no scorable tiles; skipping pair.")
            continue

        # --- Persist per-tile scores.
        csv_path = os.path.join(save_path, f'{anchor}_{partner}_fraction_overlap_tiles.csv')
        df_overlap.to_csv(csv_path, index=False)

        # --- Stats: replicate-level (primary) + per-tile descriptive + CV.
        repl, agg = replicate_level_test(df_clean)
        tile_p = tile_level_wilcoxon(df_clean)
        worst_cv, cv_map = cv_per_condition(df_clean)
        agg.to_csv(os.path.join(save_path, f'{anchor}_{partner}_replicate_means.csv'), index=False)

        tile_mean_real = float(df_clean['fraction_overlap'].mean())
        tile_mean_rot  = float(df_clean['fraction_overlap_rotated'].mean())
        logger.info(f"  per-tile mean: real={tile_mean_real:.4f} rotated={tile_mean_rot:.4f} "
                    f"delta={tile_mean_real - tile_mean_rot:+.4f} | tile Wilcoxon p={tile_p:.3g}")
        logger.info(f"  replicate ({repl['n_repl_units']} units, {repl['n_batches']} batches): "
                    f"real={repl['repl_mean_real']:.4f} rotated={repl['repl_mean_rotated']:.4f} "
                    f"delta={repl['repl_mean_delta']:+.4f} | Wilcoxon p={repl['repl_wilcoxon_p']:.3g} "
                    f"sign p={repl['repl_sign_p']:.3g}")
        logger.info(f"  CV per condition: " +
                    ", ".join(f"{c}={v:.3f}" for c, v in cv_map.items()))

        # --- Plots.
        plot_pair(df_clean, agg, pair, save_path)
        if not args.no_sample_tiles:
            sample_dir = os.path.join(save_path, 'sample_tiles')
            os.makedirs(sample_dir, exist_ok=True)
            show_sample_tiles(df_overlap, anchor, partner, MASK_K_MAP,
                              n_per_condition=2, output_dir=sample_dir)

        summary_rows.append({
            'control': control, 'panel': pair['panel'], 'anchor': anchor, 'partner': partner,
            'metric': f'P({partner}+|{anchor}+)', 'thresholds': tag,
            'n_tiles_scored': n_scored, 'n_tiles_total': n_total,
            'tile_mean_real': tile_mean_real, 'tile_mean_rotated': tile_mean_rot,
            'tile_mean_delta': tile_mean_real - tile_mean_rot, 'tile_wilcoxon_p': tile_p,
            **repl, 'worst_cv': worst_cv, 'rationale': pair['rationale'],
        })

    # --- Combined summary across all pairs.
    if not summary_rows:
        logger.error('No pairs produced results. Nothing to summarize.')
        return
    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_ROOT, 'combined_summary.csv')
    summary.to_csv(summary_csv, index=False)
    plot_combined(summary, OUTPUT_ROOT)

    logger.info('=' * 70)
    logger.info('COMBINED SUMMARY (primary test = replicate-level Wilcoxon, real > rotated):')
    show_cols = ['control', 'anchor', 'partner', 'panel', 'tile_mean_real',
                 'tile_mean_rotated', 'tile_mean_delta', 'repl_wilcoxon_p', 'worst_cv']
    logger.info('\n' + summary[show_cols].to_string(index=False))
    logger.info(f'Saved combined summary to {summary_csv}')
    logger.info(f'Log written to {log_path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()

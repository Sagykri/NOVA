import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from manuscript.plot_config import PlotConfig
from manuscript.manuscript_plot_config import *
from src.datasets.label_utils import MapLabelsFunction
from NOVA.manuscript.manuscript_figures_data_config_RANBP17_exp import (
    RANBP17_exp_ALL_CONDITIONS,
)


def _categorical_palette(items, cmap_name='hsv'):
    """Return {item: '#hex'} pulling N evenly-spaced colors from a colormap."""
    n = max(len(items), 2)
    cmap = plt.get_cmap(cmap_name, n)
    return {item: mcolors.to_hex(cmap(i)) for i, item in enumerate(items)}


class RANBP17_exp_BasePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_RANBP17_EXP_REPS = {
            'rep1': {self.MAPPINGS_ALIAS_KEY: 'Rep1', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'rep2': {self.MAPPINGS_ALIAS_KEY: 'Rep2', self.MAPPINGS_COLOR_KEY: '#4343FE'},
            'rep3': {self.MAPPINGS_ALIAS_KEY: 'Rep3', self.MAPPINGS_COLOR_KEY: '#409A14'},
            'rep4': {self.MAPPINGS_ALIAS_KEY: 'Rep4', self.MAPPINGS_COLOR_KEY: '#F2B705'},
        }

        self.COLOR_MAPPINGS_RANBP17_EXP_BATCHES = {
            'batch1': {self.MAPPINGS_ALIAS_KEY: 'Batch1', self.MAPPINGS_COLOR_KEY: '#409A14'},
        }

        self.COLOR_MAPPINGS_RANBP17_EXP_CELL_LINES = {
            'iW11': {self.MAPPINGS_ALIAS_KEY: 'iW11', self.MAPPINGS_COLOR_KEY: '#236CD9'},
        }

        self.COLOR_MAPPINGS_RANBP17_EXP_MARKERS = {
            'DAPI':    {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: '#7181C7'},
            'TDP-43':  {self.MAPPINGS_ALIAS_KEY: 'TDP-43',  self.MAPPINGS_COLOR_KEY: '#C620D2'},
            'RANBP17': {self.MAPPINGS_ALIAS_KEY: 'RANBP17', self.MAPPINGS_COLOR_KEY: '#F49DD2'},
        }

        # Hand-picked, visually-distinct colors for the 6 conditions.
        # NOTE: condition folder names use hyphens (NOT underscores) — NOVA's
        # label parser splits on '_' and would truncate underscored conditions.
        self.COLOR_MAPPINGS_RANBP17_EXP_CONDITIONS = {
            'untreated':   {self.MAPPINGS_ALIAS_KEY: 'untreated',   self.MAPPINGS_COLOR_KEY: '#7F7F7F'},
            'control-179': {self.MAPPINGS_ALIAS_KEY: 'control-179', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'control-180': {self.MAPPINGS_ALIAS_KEY: 'control-180', self.MAPPINGS_COLOR_KEY: '#17BECF'},
            'tardp-kd':    {self.MAPPINGS_ALIAS_KEY: 'tardp-kd',    self.MAPPINGS_COLOR_KEY: '#D62728'},
            'ranbp17-kd':  {self.MAPPINGS_ALIAS_KEY: 'ranbp17-kd',  self.MAPPINGS_COLOR_KEY: '#FF7F0E'},
            'both-kd':     {self.MAPPINGS_ALIAS_KEY: 'both-kd',     self.MAPPINGS_COLOR_KEY: '#9467BD'},
        }

        cell_cond_keys = [f'iW11_{c}' for c in RANBP17_exp_ALL_CONDITIONS]
        cell_cond_palette = _categorical_palette(cell_cond_keys, 'hsv')
        self.COLOR_MAPPINGS_RANBP17_EXP_CELL_LINE_CONDITIONS = {
            key: {self.MAPPINGS_ALIAS_KEY: key, self.MAPPINGS_COLOR_KEY: hex_color}
            for key, hex_color in cell_cond_palette.items()
        }

    def make_condition_palette(self, conditions, cmap_name='hsv'):
        """Build a fresh {condition: {alias, color}} dict sized to N=len(conditions).
        Falls back to the hand-picked palette where a condition is known."""
        palette = _categorical_palette(list(conditions), cmap_name)
        out = {}
        for c, hex_color in palette.items():
            if c in self.COLOR_MAPPINGS_RANBP17_EXP_CONDITIONS:
                out[c] = self.COLOR_MAPPINGS_RANBP17_EXP_CONDITIONS[c]
            else:
                out[c] = {self.MAPPINGS_ALIAS_KEY: c, self.MAPPINGS_COLOR_KEY: hex_color}
        return out

    def make_cell_line_condition_palette(self, cell_lines, conditions, cmap_name='hsv'):
        """Same idea, for the cross-product {cell_line}_{condition} keys."""
        keys = [f'{cl}_{c}' for cl in cell_lines for c in conditions]
        palette = _categorical_palette(keys, cmap_name)
        return {
            k: {self.MAPPINGS_ALIAS_KEY: k, self.MAPPINGS_COLOR_KEY: hex_color}
            for k, hex_color in palette.items()
        }

import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from manuscript.plot_config import PlotConfig
from manuscript.manuscript_plot_config import *
from src.datasets.label_utils import MapLabelsFunction
from NOVA.manuscript.FuNOVA_Screen_Conditions_Lists import (
    plate1_conditions, plate2_conditions, plate3_conditions, plate4_conditions,
)


def _categorical_palette(items, cmap_name='hsv'):
    """Return {item: '#hex'} pulling N evenly-spaced colors from a colormap."""
    n = max(len(items), 2)
    cmap = plt.get_cmap(cmap_name, n)
    return {item: mcolors.to_hex(cmap(i)) for i, item in enumerate(items)}


class FuNOVA_Screen_BasePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_FUNOVA_SCREEN_REPS = {
            'rep1': {self.MAPPINGS_ALIAS_KEY: 'Rep1', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'rep2': {self.MAPPINGS_ALIAS_KEY: 'Rep2', self.MAPPINGS_COLOR_KEY: '#4343FE'},
        }

        self.COLOR_MAPPINGS_FUNOVA_SCREEN_BATCHES = {
            'batch1': {self.MAPPINGS_ALIAS_KEY: 'Batch1', self.MAPPINGS_COLOR_KEY: '#409A14'},
        }

        self.COLOR_MAPPINGS_FUNOVA_SCREEN_CELL_LINES = {
            'C9': {self.MAPPINGS_ALIAS_KEY: 'C9', self.MAPPINGS_COLOR_KEY: '#236CD9'},
        }

        # Markers (11 + DAPI). Note: 'Aggreagtes' typo preserved to match the data.
        self.COLOR_MAPPINGS_FUNOVA_SCREEN_MARKERS = {
            'DAPI':         {self.MAPPINGS_ALIAS_KEY: 'Nucleus',      self.MAPPINGS_COLOR_KEY: '#7181C7'},
            'TDP-43':       {self.MAPPINGS_ALIAS_KEY: 'TDP-43',       self.MAPPINGS_COLOR_KEY: '#C620D2'},
            'p62':          {self.MAPPINGS_ALIAS_KEY: 'p62',          self.MAPPINGS_COLOR_KEY: '#916706'},
            'pTDP-43':      {self.MAPPINGS_ALIAS_KEY: 'pTDP-43',      self.MAPPINGS_COLOR_KEY: '#8825E5'},
            'ATF6':         {self.MAPPINGS_ALIAS_KEY: 'ATF6',         self.MAPPINGS_COLOR_KEY: '#AC166E'},
            'pAMPK':        {self.MAPPINGS_ALIAS_KEY: 'pAMPK',        self.MAPPINGS_COLOR_KEY: '#F49DD2'},
            'G3BP1':        {self.MAPPINGS_ALIAS_KEY: 'G3BP1',        self.MAPPINGS_COLOR_KEY: '#FE3B14'},
            'Calreticulin': {self.MAPPINGS_ALIAS_KEY: 'Calreticulin', self.MAPPINGS_COLOR_KEY: 'gray'},
            'Aggreagtes':   {self.MAPPINGS_ALIAS_KEY: 'Aggreagtes',   self.MAPPINGS_COLOR_KEY: '#FD0B0B'},
            'Cas3':         {self.MAPPINGS_ALIAS_KEY: 'Cas3',         self.MAPPINGS_COLOR_KEY: '#3030AC'},
            'pS6':          {self.MAPPINGS_ALIAS_KEY: 'pS6',          self.MAPPINGS_COLOR_KEY: '#FBA401'},
        }

        all_conditions = (
            plate1_conditions + plate2_conditions + plate3_conditions + plate4_conditions
        )


        cond_palette = _categorical_palette(all_conditions, 'hsv')
        self.COLOR_MAPPINGS_FUNOVA_SCREEN_CONDITIONS = {
            cond: {self.MAPPINGS_ALIAS_KEY: cond, self.MAPPINGS_COLOR_KEY: hex_color}
            for cond, hex_color in cond_palette.items()
        }

        cell_cond_palette = _categorical_palette(
            [f'C9_{c}' for c in all_conditions], 'hsv',
        )
        self.COLOR_MAPPINGS_FUNOVA_SCREEN_CELL_LINE_CONDITIONS = {
            key: {self.MAPPINGS_ALIAS_KEY: key, self.MAPPINGS_COLOR_KEY: hex_color}
            for key, hex_color in cell_cond_palette.items()
        }

    def make_condition_palette(self, conditions, cmap_name='hsv'):
        """Build a fresh {condition: {alias, color}} dict sized to N=len(conditions),
        so a UMAP showing few conditions gets visually-distinct colors instead
        of a few near-identical hues sliced from the 192-condition global palette.
        """
        palette = _categorical_palette(list(conditions), cmap_name)
        return {
            c: {self.MAPPINGS_ALIAS_KEY: c, self.MAPPINGS_COLOR_KEY: hex_color}
            for c, hex_color in palette.items()
        }

    def make_cell_line_condition_palette(self, cell_lines, conditions, cmap_name='hsv'):
        """Same idea, for the cross-product {cell_line}_{condition} keys."""
        keys = [f'{cl}_{c}' for cl in cell_lines for c in conditions]
        palette = _categorical_palette(keys, cmap_name)
        return {
            k: {self.MAPPINGS_ALIAS_KEY: k, self.MAPPINGS_COLOR_KEY: hex_color}
            for k, hex_color in palette.items()
        }

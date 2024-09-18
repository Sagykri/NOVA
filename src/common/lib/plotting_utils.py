import logging
import os

FONT_PATH = '/home/labs/hornsteinlab/sagyk/anaconda3/envs/momapsD/fonts/arial.ttf'
FONT_SIZE = 7
TITLE_SIZE = 8
AXES_LABEL_SIZE = 8
LEGEND_SIZE = 6
XTICK_LABEL_SIZE = 6
YTICK_LABEL_SIZE = 6

def save_plot(fig, savepath: str, dpi: int, save_png:bool=True, save_eps:bool=False) -> None:
    """Saves the plot if a savepath is provided, otherwise shows the plot."""
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    logging.info(f"Saving plot to {savepath}")
    if save_png:
        fig.savefig(f"{savepath}.png", dpi=dpi, bbox_inches='tight')
    if save_eps:
        fig.savefig(f"{savepath}.eps", dpi=dpi, format='eps')
    if not save_eps and not save_png:
        logging.info(f"save_eps and save_png are both False, not saving!")
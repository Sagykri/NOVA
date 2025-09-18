import logging
import os

FONT_PATH = '/home/projects/hornsteinlab/sagyk/anaconda3/envs/nova/fonts/arial.ttf'
FONT_SIZE = 7
TITLE_SIZE = 8
AXES_LABEL_SIZE = 8
LEGEND_SIZE = 6
XTICK_LABEL_SIZE = 6
YTICK_LABEL_SIZE = 6

def save_plot(fig, savepath: str, dpi: int, save_png:bool=True, save_eps:bool=False, save_pdf:bool=False) -> None:
    """Saves the plot if a savepath is provided, otherwise shows the plot."""
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    logging.info(f"Saving plot to {savepath}")
    if save_png:
        fig.savefig(f"{savepath}.png", dpi=dpi, bbox_inches='tight')
    if save_eps:
        fig.savefig(f"{savepath}.eps", dpi=dpi, bbox_inches='tight', format='eps')
    if save_pdf:
        fig.savefig(f"{savepath}.pdf", dpi=dpi, bbox_inches='tight', format='pdf')
    if not save_eps and not save_png and not save_pdf:
        logging.info(f"save_eps and save_png and save_pdf are all False, not saving!")

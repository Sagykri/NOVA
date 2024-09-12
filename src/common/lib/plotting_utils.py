import logging
import os


def save_plot(fig, savepath: str, dpi: int, save_png:bool=True, save_eps:bool=False) -> None:
    """Saves the plot if a savepath is provided, otherwise shows the plot."""
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    logging.info(f"Saving plot to {savepath}")
    if save_png:
        fig.savefig(f"{savepath}.png", dpi=dpi, bbox_inches='tight')
    elif save_eps:
        fig.savefig(f"{savepath}.eps", dpi=dpi, format='eps')
    else:
        logging.info(f"save_eps and save_png are both False, not saving!")
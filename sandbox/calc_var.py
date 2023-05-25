import logging
import os
import sys
import numpy as np

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.utils import get_training_variance, init_logging


logs_folder = os.path.join(os.getenv("MOMAPS_HOME"), "sandbox", "logs")

if __name__ == "__main__":
    init_logging(os.path.join(logs_folder, "var.log"))
    var = get_training_variance()
    logging.info(var)
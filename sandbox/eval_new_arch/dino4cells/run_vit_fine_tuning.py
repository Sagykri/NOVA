import json
import logging
import os
import sys
import importlib.util
import os
import datetime


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")


from sandbox.eval_new_arch.dino4cells.main_vit_fine_tuning import train_vit_contrastive
from sandbox.eval_new_arch.dino4cells.utils.config_utils import DictToObject


if __name__ == "__main__":    
    print("Calling the training func...")
    config_data_path = sys.argv[1]
    config_path = sys.argv[2]
    try:
        train_vit_contrastive(config_path, config_data_path)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

# ./bash_commands/run_py.sh ./sandbox/eval_new_arch/dino4cells/run_vit_fine_tuning -g -m 40000 -b 44  -a ./src/datasets/configs/training_data_config/B78NoDSTrainDatasetConfig ./sandbox/eval_new_arch/dino4cells/configs/config_finetuning/base -j finetuning -q gsla_high_gpu
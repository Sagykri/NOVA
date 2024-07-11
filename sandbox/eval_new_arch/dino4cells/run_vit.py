import logging
import os
import sys

import datetime


# os.environ['MOMAPS_HOME'] = '/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps'
# os.environ['MOMAPS_DATA_HOME'] = '/home/labs/hornsteinlab/Collaboration/MOmaps/input'

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")



# from src.common.lib.model import Model

from sandbox.eval_new_arch.dino4cells.main_vit import train_vit

class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

now_formatted = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")

config = {
        'seed': 1,
        'embedding': {
            'image_size': 100
        },
        'patch_size': 14,
        'num_channels': 2,
        
        'epochs': 300,
        
        'lr':0.0008,
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
    
        
        'batch_size_per_gpu': 250,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/logs",
        'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/tensorboard",
        "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_{now_formatted}"
    }


if __name__ == "__main__":    
    print("Calling the training func...")
    config_data_path = sys.argv[1]
    try:

        #Subset
        # config_data_path = './src/datasets/configs/training_data_config/B78NoDSTrainDatasetAugInPlaceSubsetConfig'

        #Full
        # config_data_path = './src/datasets/configs/training_data_config/B78NoDSTrainDatasetAugInPlaceConfig'


        train_vit(DictToObject(config), config_data_path)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

# ./bash_commands/run_py.sh ./sandbox/eval_new_arch/dino4cells/run_vit -g -m 40000 -b 44  -a ./src/datasets/configs/training_data_config/B78NoDSTrainDatasetAugInPlaceConfig -j training_NOVA_vit -q gsla_high_gpu
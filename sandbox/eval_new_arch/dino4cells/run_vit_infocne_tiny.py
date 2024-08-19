import logging
import os
import sys

import datetime


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")


from sandbox.eval_new_arch.dino4cells.main_vit_contrastive_infonce import train_vit_contrastive

class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

now_formatted = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
jobid = os.getenv('LSB_JOBID')

config = {
        'seed': 1,
        'vit_version': 'tiny',
        'embedding': {
            'image_size': 100
        },
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': 128,
        'negative_count':5,
        'epochs': 300,
        
        'lr': 0.0001, #0.0008
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
    
        
        'batch_size_per_gpu': 700,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Noam/MOmaps/sandbox/eval_new_arch/logs",
        'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Noam/MOmaps/sandbox/eval_new_arch/tensorboard",
        "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Noam/MOmaps/sandbox/eval_new_arch/checkpoints_{now_formatted}_{jobid}_with_FMRP"
    }


if __name__ == "__main__":    
    print("Calling the training func...")
    config_data_path = sys.argv[1]
    try:
        train_vit_contrastive(DictToObject(config), config_data_path)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

# ./bash_commands/run_py.sh ./sandbox/eval_new_arch/dino4cells/run_vit -g -m 40000 -b 44  -a ./src/datasets/configs/training_data_config/B78NoDSTrainDatasetAugInPlaceConfig -j training_NOVA_vit -q gsla_high_gpu
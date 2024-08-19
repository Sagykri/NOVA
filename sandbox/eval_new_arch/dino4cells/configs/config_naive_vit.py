import datetime
import os

now_formatted = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
jobid = os.getenv('LSB_JOBID')
jobname = os.getenv('LSB_JOBNAME')

def base():
    return {
            'seed': 1,
            'vit_version': 'tiny',
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
        
            
            'batch_size_per_gpu': 300,
            'num_workers': 6,
            
        
            'early_stopping_patience': 10,
            
            'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/logs",
            'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/tensorboard",
            "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_{now_formatted}_{jobid}_{jobname}"
        }
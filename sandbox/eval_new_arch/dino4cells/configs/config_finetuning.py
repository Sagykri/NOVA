
import datetime
import os

now_formatted = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
jobid = os.getenv('LSB_JOBID')

def base():
    return {
        'seed': 1,
        'vit_version': 'tiny',
        'embedding': {
            'image_size': 100
        },
        'layers_to_freeze': [],
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': 128,
        'include_head': True,
        'pretrained_model_num_classes': 1310,
        'pretrained_model_path': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_140724_184215_241686_879727_vit_tiny_aug_opencell/checkpoint_best.pth",

        'negative_count':5,
        'epochs': 300,
        
        'lr': 0.00005, #0.0008
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
    
        
        'batch_size_per_gpu': 700,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/logs",
        'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/tensorboard",
        "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/checkpoints/checkpoints_{now_formatted}_{jobid}"
    }
    
    
def freeze_least_changed_layers():
    """Freeze the layers that changed the least (<50 percentile) between the pretrained and non-freeze finetune
    """
    return {
        'seed': 1,
        'vit_version': 'tiny',
        'embedding': {
            'image_size': 100
        },
        'layers_to_freeze': ['blocks.10.norm1.bias', 'blocks.3.attn.proj.bias', 'cls_token',
 'blocks.2.mlp.fc1.bias', 'blocks.3.attn.qkv.bias',
 'blocks.10.attn.qkv.bias', 'blocks.1.mlp.fc2.bias', 'blocks.4.mlp.fc1.bias',
 'blocks.1.mlp.fc1.bias', 'blocks.11.mlp.fc1.bias', 'blocks.2.attn.qkv.bias',
 'blocks.3.mlp.fc1.bias', 'blocks.2.attn.proj.bias',
 'blocks.0.attn.proj.bias', 'blocks.0.mlp.fc2.bias', 'blocks.9.norm1.bias',
 'blocks.5.mlp.fc1.bias', 'blocks.6.attn.qkv.bias', 'blocks.6.mlp.fc1.bias',
 'blocks.1.attn.proj.bias', 'blocks.7.mlp.fc1.bias', 'blocks.9.mlp.fc1.bias',
 'blocks.1.norm2.bias', 'blocks.10.mlp.fc1.bias', 'blocks.8.mlp.fc1.bias',
 'blocks.8.norm1.bias', 'blocks.7.attn.qkv.bias', 'blocks.3.norm1.bias',
 'blocks.1.norm1.bias', 'blocks.9.attn.qkv.bias', 'blocks.5.attn.qkv.bias',
 'blocks.7.norm1.bias', 'blocks.0.norm2.bias', 'blocks.2.norm2.bias',
 'blocks.6.norm1.bias', 'blocks.4.attn.qkv.bias', 'blocks.2.norm1.bias',
 'blocks.3.norm2.bias', 'blocks.4.norm1.bias', 'blocks.5.norm2.bias',
 'blocks.6.norm2.bias', 'blocks.0.norm1.bias', 'blocks.5.norm1.bias',
 'blocks.0.attn.qkv.bias', 'blocks.4.norm2.bias', 'blocks.7.norm2.bias',
 'blocks.9.norm2.bias', 'blocks.8.norm2.bias', 'blocks.11.norm2.bias',
 'blocks.10.norm2.bias', 'blocks.10.norm1.weight', 'blocks.0.norm1.weight',
 'blocks.11.norm1.weight', 'blocks.6.norm1.weight', 'blocks.8.norm1.weight',
 'blocks.9.norm1.weight', 'blocks.0.norm2.weight', 'blocks.11.norm2.weight',
 'blocks.7.norm1.weight', 'blocks.5.norm1.weight', 'blocks.4.norm2.weight',
 'blocks.6.norm2.weight', 'blocks.3.norm2.weight', 'blocks.4.norm1.weight',
 'blocks.9.norm2.weight', 'blocks.5.norm2.weight', 'blocks.3.norm1.weight',
 'blocks.10.norm2.weight', 'blocks.7.norm2.weight', 'blocks.2.norm1.weight',
 'blocks.1.norm2.weight', 'blocks.2.norm2.weight', 'blocks.8.norm2.weight',
 'blocks.1.norm1.weight', 'norm.weight'],
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': 128,
        'include_head': True,
        'pretrained_model_num_classes': 1310,
        'pretrained_model_path': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_140724_184215_241686_879727_vit_tiny_aug_opencell/checkpoint_best.pth",

        'negative_count':5,
        'epochs': 300,
        
        'lr': 0.00005, #0.0008
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
    
        
        'batch_size_per_gpu': 750,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/logs",
        'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/tensorboard",
        "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/checkpoints/checkpoints_{now_formatted}_{jobid}"
    }
    
def freeze_mostly_changed_layers():
    """Freeze the layers that changed the most (>50 percentile) between the pretrained and non-freeze finetune
    """
    return {
        'seed': 1,
        'vit_version': 'tiny',
        'embedding': {
            'image_size': 100
        },
        'layers_to_freeze': ['blocks.0.attn.qkv.weight', 'blocks.0.attn.proj.weight', 'norm.bias',
 'blocks.1.attn.qkv.weight', 'blocks.2.attn.qkv.weight',
 'blocks.7.attn.proj.weight', 'blocks.3.mlp.fc2.weight',
 'blocks.4.mlp.fc2.weight', 'blocks.11.attn.proj.weight',
 'blocks.10.attn.proj.weight', 'blocks.8.attn.proj.weight',
 'blocks.5.attn.qkv.weight', 'blocks.2.mlp.fc2.weight',
 'blocks.3.attn.qkv.weight', 'blocks.9.attn.proj.weight',
 'blocks.6.attn.qkv.weight', 'blocks.4.attn.qkv.weight',
 'blocks.5.mlp.fc2.weight', 'blocks.1.mlp.fc2.weight',
 'blocks.6.mlp.fc2.weight', 'blocks.6.attn.proj.weight',
 'blocks.8.attn.qkv.weight', 'blocks.7.mlp.fc2.weight',
 'blocks.11.attn.qkv.weight', 'blocks.7.attn.qkv.weight',
 'blocks.9.attn.qkv.weight', 'blocks.2.mlp.fc1.weight',
 'blocks.8.mlp.fc2.weight', 'blocks.3.mlp.fc1.weight',
 'blocks.4.mlp.fc1.weight', 'blocks.1.mlp.fc1.weight',
 'blocks.6.attn.proj.bias', 'blocks.9.mlp.fc2.weight',
 'blocks.5.mlp.fc1.weight', 'blocks.5.attn.proj.weight',
 'blocks.11.mlp.fc2.weight', 'blocks.10.attn.qkv.weight',
 'blocks.6.mlp.fc1.weight', 'blocks.7.mlp.fc1.weight',
 'blocks.7.attn.proj.bias', 'blocks.11.mlp.fc2.bias',
 'blocks.11.mlp.fc1.weight', 'blocks.10.mlp.fc2.weight',
 'blocks.9.mlp.fc1.weight', 'blocks.5.mlp.fc2.bias',
 'blocks.8.mlp.fc1.weight', 'blocks.0.mlp.fc1.weight',
 'blocks.10.mlp.fc1.weight', 'blocks.0.mlp.fc2.weight',
 'blocks.1.attn.proj.weight', 'blocks.6.mlp.fc2.bias',
 'blocks.5.attn.proj.bias', 'blocks.4.attn.proj.weight',
 'blocks.4.mlp.fc2.bias', 'blocks.8.attn.proj.bias',
 'blocks.1.attn.qkv.bias', 'blocks.2.attn.proj.weight',
 'blocks.3.attn.proj.weight', 'blocks.7.mlp.fc2.bias',
 'patch_embed.proj.bias', 'blocks.4.attn.proj.bias',
 'blocks.11.attn.proj.bias', 'blocks.10.attn.proj.bias', 'pos_embed',
 'blocks.9.attn.proj.bias', 'blocks.3.mlp.fc2.bias',
 'blocks.10.mlp.fc2.bias', 'patch_embed.proj.weight',
 'blocks.8.mlp.fc2.bias', 'blocks.9.mlp.fc2.bias', 'blocks.11.attn.qkv.bias',
 'blocks.0.mlp.fc1.bias', 'blocks.8.attn.qkv.bias', 'blocks.11.norm1.bias',
 'blocks.2.mlp.fc2.bias'],
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': 128,
        'include_head': True,
        'pretrained_model_num_classes': 1310,
        'pretrained_model_path': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_140724_184215_241686_879727_vit_tiny_aug_opencell/checkpoint_best.pth",

        'negative_count':5,
        'epochs': 300,
        
        'lr': 0.00005, #0.0008
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
    
        
        'batch_size_per_gpu': 750,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/logs",
        'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/tensorboard",
        "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/checkpoints/checkpoints_{now_formatted}_{jobid}"
    }
    
def freeze_all_non_attn_layers():
    """Freeze all non attention layers/blocks
    """
    return {
        'seed': 1,
        'vit_version': 'tiny',
        'embedding': {
            'image_size': 100
        },
        'layers_to_freeze': ['cls_token',
 'pos_embed',
 'patch_embed.proj.weight',
 'patch_embed.proj.bias',
 'blocks.0.norm1.weight',
 'blocks.0.norm1.bias',
 'blocks.0.norm2.weight',
 'blocks.0.norm2.bias',
 'blocks.0.mlp.fc1.weight',
 'blocks.0.mlp.fc1.bias',
 'blocks.0.mlp.fc2.weight',
 'blocks.0.mlp.fc2.bias',
 'blocks.1.norm1.weight',
 'blocks.1.norm1.bias',
 'blocks.1.norm2.weight',
 'blocks.1.norm2.bias',
 'blocks.1.mlp.fc1.weight',
 'blocks.1.mlp.fc1.bias',
 'blocks.1.mlp.fc2.weight',
 'blocks.1.mlp.fc2.bias',
 'blocks.2.norm1.weight',
 'blocks.2.norm1.bias',
 'blocks.2.norm2.weight',
 'blocks.2.norm2.bias',
 'blocks.2.mlp.fc1.weight',
 'blocks.2.mlp.fc1.bias',
 'blocks.2.mlp.fc2.weight',
 'blocks.2.mlp.fc2.bias',
 'blocks.3.norm1.weight',
 'blocks.3.norm1.bias',
 'blocks.3.norm2.weight',
 'blocks.3.norm2.bias',
 'blocks.3.mlp.fc1.weight',
 'blocks.3.mlp.fc1.bias',
 'blocks.3.mlp.fc2.weight',
 'blocks.3.mlp.fc2.bias',
 'blocks.4.norm1.weight',
 'blocks.4.norm1.bias',
 'blocks.4.norm2.weight',
 'blocks.4.norm2.bias',
 'blocks.4.mlp.fc1.weight',
 'blocks.4.mlp.fc1.bias',
 'blocks.4.mlp.fc2.weight',
 'blocks.4.mlp.fc2.bias',
 'blocks.5.norm1.weight',
 'blocks.5.norm1.bias',
 'blocks.5.norm2.weight',
 'blocks.5.norm2.bias',
 'blocks.5.mlp.fc1.weight',
 'blocks.5.mlp.fc1.bias',
 'blocks.5.mlp.fc2.weight',
 'blocks.5.mlp.fc2.bias',
 'blocks.6.norm1.weight',
 'blocks.6.norm1.bias',
 'blocks.6.norm2.weight',
 'blocks.6.norm2.bias',
 'blocks.6.mlp.fc1.weight',
 'blocks.6.mlp.fc1.bias',
 'blocks.6.mlp.fc2.weight',
 'blocks.6.mlp.fc2.bias',
 'blocks.7.norm1.weight',
 'blocks.7.norm1.bias',
 'blocks.7.norm2.weight',
 'blocks.7.norm2.bias',
 'blocks.7.mlp.fc1.weight',
 'blocks.7.mlp.fc1.bias',
 'blocks.7.mlp.fc2.weight',
 'blocks.7.mlp.fc2.bias',
 'blocks.8.norm1.weight',
 'blocks.8.norm1.bias',
 'blocks.8.norm2.weight',
 'blocks.8.norm2.bias',
 'blocks.8.mlp.fc1.weight',
 'blocks.8.mlp.fc1.bias',
 'blocks.8.mlp.fc2.weight',
 'blocks.8.mlp.fc2.bias',
 'blocks.9.norm1.weight',
 'blocks.9.norm1.bias',
 'blocks.9.norm2.weight',
 'blocks.9.norm2.bias',
 'blocks.9.mlp.fc1.weight',
 'blocks.9.mlp.fc1.bias',
 'blocks.9.mlp.fc2.weight',
 'blocks.9.mlp.fc2.bias',
 'blocks.10.norm1.weight',
 'blocks.10.norm1.bias',
 'blocks.10.norm2.weight',
 'blocks.10.norm2.bias',
 'blocks.10.mlp.fc1.weight',
 'blocks.10.mlp.fc1.bias',
 'blocks.10.mlp.fc2.weight',
 'blocks.10.mlp.fc2.bias',
 'blocks.11.norm1.weight',
 'blocks.11.norm1.bias',
 'blocks.11.norm2.weight',
 'blocks.11.norm2.bias',
 'blocks.11.mlp.fc1.weight',
 'blocks.11.mlp.fc1.bias',
 'blocks.11.mlp.fc2.weight',
 'blocks.11.mlp.fc2.bias',
 'norm.weight',
 'norm.bias',
 'head.weight',
 'head.bias'],
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': 128,
        'include_head': True,
        'pretrained_model_num_classes': 1310,
        'pretrained_model_path': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_140724_184215_241686_879727_vit_tiny_aug_opencell/checkpoint_best.pth",

        'negative_count':5,
        'epochs': 300,
        
        'lr': 0.00005, #0.0008
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
    
        
        'batch_size_per_gpu': 750,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/logs",
        'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/tensorboard",
        "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_fine_tuned/checkpoints/checkpoints_{now_formatted}_{jobid}"
    }
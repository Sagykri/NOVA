import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging

# from src.common.lib.models.NOVA_model import NOVAModel
from src.common.lib.embeddings_utils import generate_embeddings, save_embeddings
from src.common.lib.utils import load_config_file, handle_log

#TODO:remove
from sandbox.eval_new_arch.dino4cells.archs import vision_transformer as vits
import torch
import random
import torch.backends.cudnn as cudnn
from sandbox.eval_new_arch.dino4cells.utils import utils
class DictToObject: #TODO: remove
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

def generate_embeddings_with_model(chkp_path:str, config_path_data:str)->None:
    # model = NOVAModel.load_from_checkpoint(sys.argv[1])
    # model_output_folder = model.config_trainer.OUTPUTS_FOLDER
    config = {
        'seed': 1,
        'embedding': {
            'image_size': 100
        },
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': 128, #int(sys.argv[3]),
        
        'batch_size_per_gpu': 700,#300,#3,#65,
        'num_workers': 6,  

        'vit_version':'tiny'      
    }
    config = DictToObject(config)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    cudnn.benchmark = False
    random.seed(config.seed)

    create_vit = vits.vit_base
    if config.vit_version == 'base':
        create_vit = vits.vit_base
    elif config.vit_version == 'small':
        create_vit = vits.vit_small
    elif config.vit_version == 'tiny':
        create_vit = vits.vit_tiny

    model = create_vit(
        img_size=[config.embedding.image_size],
        patch_size=config.patch_size,
        in_chans=config.num_channels,
        num_classes=config.num_classes
    ).cuda()

    model = utils.load_model_from_checkpoint(chkp_path, model)
    model_output_folder = os.sep.join(chkp_path.split(os.sep)[:-2]) #model.trainer_config.OUTPUTS_FOLDER #TODO!!
    handle_log(model_output_folder)

    config_data = load_config_file(config_path_data)

    embeddings, labels = generate_embeddings(model, config_data)
    save_embeddings(embeddings, labels, config_data, model_output_folder)

if __name__ == "__main__":
    print("Starting generate embeddings...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply trained model path (.pth) and data config.")
        chkp_path = sys.argv[1]
        config_path_data = sys.argv[2]
        generate_embeddings_with_model(chkp_path, config_path_data)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

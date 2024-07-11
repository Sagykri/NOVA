import os
import random
import sys



sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import torch

from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.dataset import Dataset
from src.common.lib.utils import load_config_file

# from sandbox.eval_new_arch.dino4cells.main_vit import infer_pass
from sandbox.eval_new_arch.dino4cells.main_vit_contrastive import infer_pass
from sandbox.eval_new_arch.dino4cells.utils import utils
from sandbox.eval_new_arch.dino4cells.archs import vision_transformer as vits

import torch.backends.cudnn as cudnn

class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

def predict_using_fc():
    
    return_cls_token = True
    
    # if len(sys.argv) < 3:
    #     raise ValueError("Invalid config path. Must supply model config and data config.")
    
    # config_path_model = sys.argv[1]
    # config = {
    #     'seed': 1,
    #     'embedding': {
    #         'image_size': 100
    #     },
    #     'patch_size': 14,
    #     'num_channels': 2,
        
    #     'epochs': 300,
        
    #     'lr':0.0008,
    #     'min_lr': 1e-6,
    #     'warmup_epochs': 5,
        
    #     'weight_decay': 0.04,
    #     'weight_decay_end': 0.4,
    
        
    #     'batch_size_per_gpu': 300,#3,#65,
    #     'num_workers': 6,
        
    #     'accumulation_steps': 1,
    
    #     'early_stopping_patience': 10,
        
    #     # 'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/logs",
    #     # 'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/tensorboard",
    #     # "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_{now_formatted}"
    # }
    
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
        'local_crops_number': None,
        
        'batch_size_per_gpu': 200,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
        # 'logs_dir':"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_contrastive/logs",
        # 'tensorboard_root_folder': "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_contrastive/tensorboard",
        # "output_dir": f"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_contrastive/checkpoints/checkpoints_{now_formatted}"
    }
    
    config = DictToObject(config)
    config_path_data = sys.argv[1]

    # model, config_model =  init_model_for_embeddings(config_path_model=config_path_model)
    config_data = load_config_file(config_path_data)
    
    logging.info('Not doing softmax!!')
    # if len(sys.argv) > 3:
    output_folder_path = sys.argv[2]
    # output_folder_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/noam_eval_fc/full_model_basic_aug_no_drop_b6rep2"

    # output_folder_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/noam_eval_fc/full_model_contrastive_with_shuffle/b9rep1_final_epoch"
    # else:
        # output_folder_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "pretext")
    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path)

    jobid = os.getenv('LSB_JOBID')
    logging.info(f"init (jobid: {jobid})")
    logging.info("[Predict label with fc]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    
    
    # utils.init_distributed_mode(args)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    # utils.fix_random_seeds(config.seed)
    # init_signal_handler()
    # logging.info("git:\n  {}\n".format(utils.get_sha()))
    # logging.info(
    #     "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    # )
    # cudnn.benchmark = True
    cudnn.benchmark = False
    random.seed(config.seed)

    # chosen_loader = file_dataset.image_modes[config["model"]["image_mode"]]
    # FileList = file_dataset.data_loaders[config["model"]["datatype"]]

    # ============ preparing data ... ============
    # transform =  DataAugmentationVIT(config=config)
    # transform = transforms.Compose([
    #         # RandomResizedCropWithCheck(100, scale=(0.6, 1.0)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip()])
    
    config_data = load_config_file(config_path_data)

    dataset = DatasetSPD(config_data, None)
    # config.out_dim = len(np.unique(dataset_train.label))
    train_indexes, val_indexes, test_indexes = None, None, None
    # dataset_val, dataset_test = None, None


    # if is_one_config_supplied:
    # dataset_val, dataset_test = deepcopy(dataset_train), deepcopy(dataset_train) # the deepcopy is important. do not change. 
    # dataset_test.flip, dataset_test.rot = False, False
    if config_data.SPLIT_DATA:
        logging.info("Split data...")
        train_indexes, val_indexes, test_indexes = dataset.split()
        dataset_test_subset = Dataset.get_subset(dataset, test_indexes)
    else:
        # dataset_val, dataset_test = DatasetSPD(config_val), DatasetSPD(config_test)
        dataset_test_subset = dataset

    
    
    
    data_loader_test = get_dataloader(dataset_test_subset, config.batch_size_per_gpu, num_workers=config.num_workers)
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    model = vits.vit_base(
            img_size=[config.embedding.image_size, config.embedding.image_size],
            patch_size=config.patch_size,
            # drop_path_rate=0.1,  # stochastic depth
            # drop_rate=0.3, # can't go together with drop_path_rate - cuda out of memory
            in_chans=config.num_channels,
            num_classes=256#len(dataset.unique_markers)
    ).cuda()
    
    # full model
    # chkp_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_190624_072039_132123_b78_no_drop_basic_aug/checkpoint_best.pth"
    # wt untreated model
    # chkp_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit/checkpoints_190624_161119_860502_no_drop_basic_aug_only_wt_untreated/checkpoint_best.pth"#
    
    
    chkp_path = sys.argv[3]#"/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/vit_contrastive/checkpoints/checkpoints_240624_232822_472218_with_shuffle/checkpoint_best.pth"
    model = utils.load_model_from_checkpoint(chkp_path, model)
    
    predictions, labels, cls_tokens = infer_pass(model, data_loader_test, return_cls_token=return_cls_token)
    set_type = 'testset'
    np.save(os.path.join(output_folder_path,f'{set_type}_pred.npy'), predictions)
    np.save(os.path.join(output_folder_path,f'{set_type}_true_labels.npy'), np.array(labels))
    if cls_tokens is not None:
        np.save(os.path.join(output_folder_path,f'{set_type}_cls_tokens.npy'), cls_tokens)

    logging.info(f'Finished {set_type} set, saved in {output_folder_path}')
    # # ****** IMPORTANT: shuffle=False !!! to help get correct tile numbers per image (avoid shuffling tiles indise a site)****** 
    # datasets_list = load_dataset_for_embeddings(config_data=config_data, batch_size=50, config_model=config_model, 
    #                                             shuffle=False)
    # logging.info(f'after [load_dataset_for_embeddings] datasets_list[0].dataset.unique_markers.shape {datasets_list[0].dataset.unique_markers.shape}')
    
    # run_multi_checkpoints = True
    # if run_multi_checkpoints:
    #     checks = sorted(os.listdir(os.path.join(config_model.MODEL_OUTPUT_FOLDER, "checkpoints")))[::2]
    #     for c in checks:
    #         logging.info(f'running checkpoint {c}')
    #         model.conf.MODEL_PATH = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'checkpoints', c)
    #         logging.info(f'model.conf.MODEL_PATH: {model.conf.MODEL_PATH}')
    #         output_folder_path = os.path.join(output_folder_path, c)
    #         logging.info(f'output_folder_path: {output_folder_path}')
    #         trained_model = load_model_with_dataloader(model, datasets_list)
    #         __predict_with_dataloader(trained_model, output_folder_path, datasets_list)
    
    # else:
    #     trained_model = load_model_with_dataloader(model, datasets_list)
    #     __predict_with_dataloader(trained_model, output_folder_path, datasets_list)

# def __predict_with_dataloader(model, output_folder_path, datasets_list):

#     # Parser to get the image's batch/cell_line/condition/rep/marker
#     def label(full_path):
#         path_list = full_path.split(os.sep)
#         cell_line_condition_marker_list = path_list[-4:-1]
#         cell_line_condition_marker = '_'.join(cell_line_condition_marker_list)
#         return cell_line_condition_marker
#     get_label = np.vectorize(label)
    
#     def do_label_prediction(images_batch, all_predictions, all_paths, all_labels):
#         # images_batch is torch.Tensor of size(n_tiles, n_channels, 100, 100)
#         paths = images_batch['image_path']
#         labels = get_label(paths)
#         # path_with_tiles = [f'{path}_{n_tile}' for n_tile, path in enumerate(images_batch['image_path'])]
#         logging.info(images_batch['image'].numpy().shape)
#         predictions = model.model.infer_embeddings(images_batch['image'].numpy(), output_layer='fc1')
#         logging.info(f'predictions shape: {predictions.shape}')
#         all_predictions.append(predictions)
#         all_paths.extend(paths)
#         all_labels.extend(labels)
#         return all_predictions, all_paths, all_labels
    
#     def predict_for_set(set_type, set_index, datasets_list):
#         logging.info(f"Predict label - {set_type} set")
#         all_predictions, all_paths, all_labels = [], [], []
#         for i, images_batch in enumerate(datasets_list[set_index]):
#             all_predictions, all_paths, all_labels = do_label_prediction(images_batch, all_predictions, all_paths, all_labels)
#         all_predictions = np.concatenate(all_predictions, axis=1)
#         logging.info(f'all_predictions shape: {all_predictions.shape}')
#         # all_predictions_softmax = softmax(all_predictions, axis=-1)
#         np.save(os.path.join(output_folder_path,f'{set_type}_pred.npy'), all_predictions)
#         np.save(os.path.join(output_folder_path,f'{set_type}_paths.npy'), np.array(all_paths))
#         np.save(os.path.join(output_folder_path,f'{set_type}_true_labels.npy'), np.array(all_labels))
#         logging.info(f'Finished {set_type} set, saved in {output_folder_path}')
#         return None
    
#     if len(datasets_list)==3:
#         predict_for_set('testset', 2 , datasets_list)
#         # predict_for_set('valset', 1 , datasets_list)
#         # predict_for_set('trainset', 0 , datasets_list)
        
#     elif len(datasets_list)==1:
#         predict_for_set('all', 0 , datasets_list)
    
#     else:
#         logging.exception("[Generate Embeddings] Load model: List of datasets is not supported.")
    
#     return None
        

if __name__ == "__main__":
    print("Starting predicting using fc...")
    try:
        predict_using_fc()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

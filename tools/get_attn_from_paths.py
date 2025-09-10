import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
import logging
from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import load_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.dataset_PATHS import DatasetFromPaths
from src.datasets.data_loader import get_dataloader
from src.figures.plot_config import PlotConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.attention_maps.attention_config import AttnConfig
from src.attention_maps.attention_maps_utils import generate_attn_maps, process_attn_maps, save_attn_maps
from src.figures.attention_maps_plotting import plot_attn_maps
from src.analysis.analyzer_attention_correlation import AnalyzerAttnCorr


# arguments: model, dataset config, attn_config, plot_attn_config 

# load configs
# load model 
# get attn matrix from model (generate using infernce)
# process attn matrix (process attn maps)
# plot - create fig 

def generate_attn_maps_with_model(paths:list, outputs_folder_path:str, config_path_data:str, 
                                config_path_attn:str, config_path_corr:str, 
                                config_path_plot:str, batch_size:int=10)->None:
    """
        For each sample in the data config - 
            - extracts the attention maps from the model 
            - saves the raw attention maps
            - process the attention maps according to the parameters in the attn config
            - saves the processed attn maps
    """

    MODEL_DIR="/home/projects/hornsteinlab/Collaboration/NOVA/outputs/vit_models"
    MODEL_NAMES=['finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen', 'pretrained_model']

    for model_name in MODEL_NAMES: 

        outputs_folder_path = os.path.join(MODEL_DIR, model_name)
        # load configs
        config_data:DatasetConfig = load_config_file(config_path_data, "data")
        config_attn:AttnConfig = load_config_file(config_path_attn, "data")
        config_data.OUTPUTS_FOLDER = outputs_folder_path
        config_corr = load_config_file(config_path_corr, "data")
        config_plot:PlotAttnMapConfig = load_config_file(config_path_plot, "plot")
        
        # load model
        chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
        model = NOVAModel.load_from_checkpoint(chkp_path)

        corr_method = config_corr.CORR_METHOD
        for description, paths in paths_by_type.items():
            temp_output_path = os.path.join("attn_by_paths", "FUS_corr_scores", corr_method, description, model_name)
            batch_size = len(paths)
            config_data.MARKERS = [ 'ANAX11',
                                    'Calreticulin',
                                    'CD41',
                                    'CLTC',
                                    'DAPI',
                                    'DCP1A',
                                    'FMRP',
                                    'FUS',
                                    'G3BP1',
                                    'GM130',
                                    'KIF5A',
                                    'LAMP1',
                                    'MitoTracker',
                                    'NCL',
                                    'NEMO',
                                    'P54',
                                    'PEX14',
                                    'Phalloidin',
                                    'PML',
                                    'PSD95',
                                    'PURA',
                                    'SNCA',
                                    'SQSTM1',
                                    'TDP43',
                                    'TIA1',
                                    'TOMM20',
                                    'TUJ1']
            # create dataset
            dataset = DatasetFromPaths(config_data, paths)
            
            # generate (extract from model) raw attention maps and save
            attn_maps, labels, paths = __generate_attn_maps_with_paths_dataloader(
                dataset=dataset, model=model, batch_size=batch_size, num_workers=1)

            # process the raw attn_map and save 
            processed_attn_maps = process_attn_maps([attn_maps], [labels], config_data, config_attn)

            d = AnalyzerAttnCorr(config_data, outputs_folder_path, config_corr)
            corr_data = d.calculate(processed_attn_maps, [labels], [paths])

            plot_attn_maps(processed_attn_maps, [labels], [paths], config_data, config_plot, output_folder_path=temp_output_path, num_workers = 2, corr_data =  corr_data,corr_method = corr_method)



def __generate_attn_maps_with_paths_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths

paths_by_type = {
        "NIH_B1_FUSHetro_UT":
        [
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/ANAX11/rep1_s14_panelD_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/ANAX11/rep8_s13_panelD_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/Calreticulin/rep1_s12_panelH_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/Calreticulin/rep4_s1_panelH_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/CD41/rep2_s25_panelJ_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/CD41/rep8_s3_panelJ_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/CLTC/rep1_s22_panelB_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/CLTC/rep5_s24_panelB_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/DCP1A/rep1_s12_panelC_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/DCP1A/rep2_s20_panelC_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/FMRP/rep1_s14_panelA_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/FMRP/rep4_s1_panelA_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/FUS/rep1_s17_panelK_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/FUS/rep4_s12_panelK_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/G3BP1/rep1_s18_panelG_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/G3BP1/rep4_s16_panelG_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/GM130/rep1_s9_panelI_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/GM130/rep2_s22_panelI_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/KIF5A/rep3_s18_panelG_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/KIF5A/rep7_s9_panelG_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/LAMP1/rep1_s10_panelH_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/LAMP1/rep8_s12_panelH_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/MitoTracker/rep2_s22_panelK_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/MitoTracker/rep3_s16_panelK_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/NCL/rep1_s22_panelK_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/NCL/rep8_s8_panelK_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/TDP43/rep1_s9_panelJ_FUSHeterozygous_processed.npy",
            "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/NIH/batch1/FUSHeterozygous/Untreated/TDP43/rep4_s21_panelJ_FUSHeterozygous_processed.npy"
        ]

    }

if __name__ == "__main__":
    print("Starting generate attention maps...")
    

    try:
        if len(sys.argv) < 6:
            raise ValueError("Invalid arguments. Must supply model path, data config, attn config, plot_attn_config")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]
        config_path_attn = sys.argv[3]
        config_path_corr = sys.argv[4]
        config_path_plot = sys.argv[5]

        if len(sys.argv)==7:
            try:
                batch_size = int(sys.argv[6])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 10
        generate_attn_maps_with_model(paths_by_type, outputs_folder_path, config_path_data, 
                                    config_path_attn, config_path_corr,
                                    config_path_plot, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

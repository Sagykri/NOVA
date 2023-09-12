
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")
from src.common.lib.utils import getfreegpumem, init_logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################################################
#|                                                        |
#|                                                        |
#|                                                        |
#|      ░█████╗░░█████╗░███╗░░██╗███████╗██╗░██████╗░     |
#|      ██╔══██╗██╔══██╗████╗░██║██╔════╝██║██╔════╝░     | 
#|      ██║░░╚═╝██║░░██║██╔██╗██║█████╗░░██║██║░░██╗░     |
#|      ██║░░██╗██║░░██║██║╚████║██╔══╝░░██║██║░░╚██╗     |
#|      ╚█████╔╝╚█████╔╝██║░╚███║██║░░░░░██║╚██████╔╝     |
#|      ░╚════╝░░╚════╝░╚═╝░░╚══╝╚═╝░░░░░╚═╝░╚═════╝░     |
#|                                                        |
#|                                                        |
###########################################################


log_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_ae_log.log"
model_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/ae_model.pt"
embeddings_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch78_nods_tl_ep23/features/sm_embeddings_b5_fix_vqvec1_test.npy"
savepath = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/sm_embeddings_b5_fix_vqvec1_test_reduced.npy"



###########################################################
#|                                                        |
#|                                                        |
#|                                                        |
#|  ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░░██████╗   |
#|  ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░██╔════╝   |
#|  ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░╚█████╗░   |
#|  ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░░╚═══██╗   |
#|  ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗██████╔╝   |
#|  ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝╚═════╝░   |
#|                                                        |
#|                                                        |
###########################################################

class Conv3DAutoencoder(nn.Module):
    # Assume `your_data` is of shape [batch_size, 25, 64, 25, 25]
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(25, 50, kernel_size=4, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(50, 32, kernel_size=3, stride=3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 50, kernel_size=3, stride=3, padding=1, output_padding=(2,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(50, 25, kernel_size=4, stride=3, padding=1, output_padding=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        logging.info(f"Encoder output shape: {x.shape}")
        x = self.decoder(x)
        return x

#########################################################################################
#|                                                                                      |
#|                                                                                      |
#|                                                                                      |
#|     ███████╗██╗░░░██╗███╗░░██╗░█████╗░████████╗██╗░█████╗░███╗░░██╗░██████╗          |
#|     ██╔════╝██║░░░██║████╗░██║██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║██╔════╝          |
#|     █████╗░░██║░░░██║██╔██╗██║██║░░╚═╝░░░██║░░░██║██║░░██║██╔██╗██║╚█████╗░          |
#|     ██╔══╝░░██║░░░██║██║╚████║██║░░██╗░░░██║░░░██║██║░░██║██║╚████║░╚═══██╗          |
#|     ██║░░░░░╚██████╔╝██║░╚███║╚█████╔╝░░░██║░░░██║╚█████╔╝██║░╚███║██████╔╝          |
#|     ╚═╝░░░░░░╚═════╝░╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝╚═════╝░          |
#|                                                                                      |
#|                                                                                      |
#########################################################################################
        


def __load_model(model_path):
    logging.info(f"Loading model from {model_path}...")
    
    model = Conv3DAutoencoder()
    model.load_state_dict(torch.load(model_path))
    
    return model

def run_eval_ae():
    logging.info(f"Device: {device}")
    logging.info(getfreegpumem(0))
    
    logging.info(f"Loading data: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logging.info("Converting numpy to tensor")
    embeddings = torch.from_numpy(embeddings)
    
    logging.info(f"embeddings.shape = {embeddings.shape}")
    logging.info("Reshaping embeddings into [batch_size, #markers, 64, 25 ,25]")
    
    embeddings = embeddings.reshape(embeddings.shape[0], -1, 64, 25, 25)
    
    logging.info(f"embeddings.shape = {embeddings.shape}")
    
    
    model = __load_model(model_path)
    
    logging.info("Eval..")
    model.eval()
    with torch.no_grad():
        reduced_data = model.encoder(embeddings)
        
        logging.info(f"reduced_data shape= {reduced_data.shape}")
        
        logging.info(f"Saving reduced_data to file: {savepath}")
        with open(savepath, 'wb') as f:
            np.save(f, reduced_data)
        
        embeddings_reconstructed = model.decoder(reduced_data)
        mse = torch.nn.functional.mse_loss(embeddings, embeddings_reconstructed)
        logging.info(f"MSE = {mse}")
    

if __name__ == "__main__":
    init_logging(log_path)
    
    logging.info("Running eval ae...")
    try:
        run_eval_ae()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")



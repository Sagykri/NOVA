
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
#|  ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░░██████╗   |
#|  ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░██╔════╝   |
#|  ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░╚█████╗░   |
#|  ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░░╚═══██╗   |
#|  ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗██████╔╝   |
#|  ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝╚═════╝░   |
#|                                                        |
#|                                                        |
###########################################################


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100000000, 10000, device=device),
            nn.ReLU(),
            nn.Linear(10000, 8000, device=device),
            nn.ReLU(),
            nn.Linear(8000, 5000, device=device)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5000, 8000, device=device),
            nn.ReLU(),
            nn.Linear(8000, 10000, device=device),
            nn.ReLU(),
            nn.Linear(10000, 100000000, device=device)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class Conv1DAutoencoder(nn.Module):
    # Assume `your_data` is of shape [batch_size, 25, 40000]
    def __init__(self):
        super(Conv1DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(50, 100, kernel_size=5, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(100, 50, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(50, 25, kernel_size=5, stride=2, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

def __train_ae(X_train, X_val):
    
    epochs = 100
    early_stop_patience = 5
    batch_size = 32
    
    
    early_stop_counter = 0
    best_val_loss = np.inf
    
    # Initialize the autoencoder and the optimizer
    logging.info("Initiating 3D Autoencoder...")
    model = Conv3DAutoencoder() #Autoencoder()
    logging.info(f"Moving model to {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        logging.info(f"{epoch}/{epochs}")
        
        logging.info("----- Training phase ------")
        for batch in range(0, len(X_train), batch_size):
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_train)//batch_size}, Train]")
            
            logging.info(getfreegpumem(0))
            
            X_train_batch = X_train[batch:batch+batch_size]
            
            logging.info(f"Moving data to {device}")
            X_train_batch = X_train_batch.to(device)
            
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_train)//batch_size}, Train] X_train_batch.shape: {X_train_batch.shape}")
            
            model.train()
            
            optimizer.zero_grad()
            
            # Forward pass
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_train)//batch_size}, Train] Forward")
            outputs = model(X_train_batch)
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_train)//batch_size}, Train] outputs shape: {outputs.shape}")
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_train)//batch_size}, Train] Calc loss")
            loss = criterion(outputs, X_train_batch)
            
            # Backward pass and optimization
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_train)//batch_size}, Train] backward and step")
            loss.backward()
            optimizer.step()
            
        logging.info("----- Eval phase ------")
        best_val_loss_ep = np.inf
        for batch in range(0, len(X_val), batch_size):
            logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_val)//batch_size}, Eval]")
            with torch.inference_mode():
                model.eval()
                
                X_val_batch = X_val[batch:batch+batch_size]
                logging.info(f"Moving data to {device}")
                X_val_batch = X_val_batch.to(device)
                
                logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_val)//batch_size}, Eval] X_val_batch.shape: {X_val_batch.shape}")
                
                logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_val)//batch_size}, Eval] Forward eval")
                outputs_val = model(X_val_batch)
                logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_val)//batch_size}, Eval] Calc loss val")
                loss_val = criterion(outputs_val, X_val_batch)
                logging.info(f"[E{epoch}/{epochs}, B{batch//batch_size}/{len(X_val)//batch_size}, Eval] Val Loss: {loss_val}")
                
                if loss_val < best_val_loss_ep:
                    best_val_loss_ep = loss_val
                
        if best_val_loss_ep < best_val_loss:
            early_stop_counter = 0
            logging.info(f"[E{epoch}/{epochs}] New best loss! {best_val_loss_ep} (previous: {best_val_loss})")
            best_val_loss = best_val_loss_ep
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop_patience:
            logging.info(f"[E{epoch}/{epochs}] Reached early stopping ({early_stop_patience})")
            break
                
        # Free mem - empty cache
        logging.info(f"[E{epoch}/{epochs}] Empty cache")
        torch.cuda.empty_cache()
        
    return model
        

# # Get the reduced dimension data
# with torch.no_grad():
#     reduced_data = model.encoder(your_data)

def __save_model(model):
    fpath = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/ae_model.pt"
    logging.info(f"Saving model to {fpath}...")
    torch.save(model.state_dict(), fpath)


def run_train_ae():
    logging.info(f"Device: {device}")
    logging.info(getfreegpumem(0))
    
    logging.info("Loading data")
    embeddings = np.load("/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch78_nods_tl_ep23/features/sm_embeddings_b9_fix_vqvec1.npy")
    embeddings_test = np.load("/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch78_nods_tl_ep23/features/sm_embeddings_b9_fix_vqvec1_test.npy")
    
    embeddings = torch.from_numpy(embeddings)
    embeddings_test = torch.from_numpy(embeddings_test)
    
    logging.info(f"embeddings.shape = {embeddings.shape}, embeddings_test.shape = {embeddings_test.shape}")
    logging.info("Reshaping embeddings into [batch_size, #markers, 64, 25 ,25]")
    
    embeddings = embeddings.reshape(embeddings.shape[0], -1, 64, 25, 25)
    embeddings_test = embeddings_test.reshape(embeddings_test.shape[0], -1, 64, 25, 25)
    
    logging.info(f"embeddings.shape = {embeddings.shape}, embeddings_test.shape = {embeddings_test.shape}")
    
    
    model = __train_ae(embeddings, embeddings_test)
    
    __save_model(model)

if __name__ == "__main__":
    init_logging("/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/train_ae_log.log")
    
    logging.info("Running train ae...")
    try:
        run_train_ae()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")



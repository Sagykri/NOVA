import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import logging

from sklearn.model_selection import train_test_split

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from sandbox.siamese_net.contrastive_loss import ContrastiveLoss
from sandbox.siamese_net.siamese_network import SiameseNetwork
from sandbox.siamese_net.utils import train_siamese_network, get_shape_npy, init_logging
from sandbox.siamese_net.dataset import SiameseDataset

def train():
    init_logging("/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/siamese_net/logs/log_train.txt")

    logging.info("INIT")
    logging.info("[Training]")

    # Params
    num_epochs = 10
    batch_size = 32
    early_stopping_patience = 5
    # input_dim = 9216
    learning_rate = 0.001
    SEED = 1

    # Initialize the Siamese network and the contrastive loss
    logging.info("Initiating model and loss function")
    model = SiameseNetwork()
    loss = ContrastiveLoss(margin=2.0)


    # Use an optimizer like SGD or Adam
    logging.info("Init optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloaders
    logging.info("Init datasets and dataloaders")
    data_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/sm_embeddings_b9_new.npy"
    labels_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/sm_labels_b9_new.npy"
    logging.info("Get shape")
    data_shape = np.load(data_path,mmap_mode='r').shape#get_shape_npy(data_path)
    logging.info(f"Data shape: {data_shape}")
    logging.info("Split to train and val set")
    train_indexes, val_indexes = train_test_split(np.arange(data_shape[0]),
                                                train_size=0.8,
                                                shuffle=True,
                                                random_state=SEED)
    logging.info(f"Init datasets for train and val (train: {len(train_indexes)}, val: {len(val_indexes)})")
    train_dataset, val_dataset = SiameseDataset(data_path, labels_path, train_indexes),\
                                    SiameseDataset(data_path, labels_path, val_indexes)
    logging.info("Init dataloaders")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)


    # Train
    model_save_folder = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/siamese_net/checkpoints"
    logging.info(f"Train (savefolder={model_save_folder})")
    train_siamese_network(model, loss, train_dataloader, val_dataloader,
                        optimizer, num_epochs=num_epochs,
                        early_stopping_patience=early_stopping_patience,
                        save_folder=model_save_folder)



if __name__ == "__main__":    
    print("Calling the training func...")
    try:
        train()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

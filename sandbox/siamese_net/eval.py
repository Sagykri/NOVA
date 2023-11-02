import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import numpy as np
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from sandbox.siamese_net.siamese_network import SiameseNetwork
from sandbox.siamese_net.utils import load_checkpoint, obtain_embeddings, plot_embeddings_umap, init_logging
from sandbox.siamese_net.dataset import SiameseDataset
from torch.utils.data import DataLoader
import logging

def eval_model():
    init_logging("/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/siamese_net/logs/log_eval.txt")

    logging.info("Init")
    logging.info("[Eval]")

    # Params
    model_filepath = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/siamese_net/checkpoints/checkpoint_3.pth"
    # input_dim = 9216
    learning_rate = 0.001
    batch_size = 32

    # Initialize the Siamese network and the contrastive loss
    logging.info("Init model")
    model = SiameseNetwork()

    # Use an optimizer like SGD or Adam
    logging.info("Init optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloader
    logging.info("Init dataset")
    data_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/sm_embeddings_b9_new.npy"
    labels_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/sm_labels_b9_new.npy"
    test_dataset = SiameseDataset(data_path, labels_path)
    unique_markers = test_dataset.unique_markers
    logging.info("Init dataloader")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

    # Load checkpoint
    logging.info(f"Load checkpoint ({model_filepath})")
    load_checkpoint(model, optimizer, file_path=model_filepath)

    # Eval
    logging.info(f"Obtain embeddings")
    embeddings, labels = obtain_embeddings(model, test_dataloader)
    logging.info(f"Embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")
    logging.info(f"labels unique: {np.unique(labels)}, sample: {labels[:10]}")
    logging.info(f"Generating umap")
    plot_embeddings_umap(embeddings, labels, unique_markers, savepath="/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/siamese_net/plots/fig.png")


if __name__ == "__main__":    
    print("Calling the eval func...")
    try:
        eval_model()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import umap
import matplotlib.pyplot as plt
import numpy as np


def save_checkpoint(epoch, model, optimizer, file_path="checkpoint.pth"):
    """Save a training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    
def load_checkpoint(model, optimizer, file_path="checkpoint.pth"):
    """Load a training checkpoint and restore the model and optimizer states"""
    if os.path.isfile(file_path):
        logging.info(f"=> Loading checkpoint '{file_path}'")
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"=> Loaded checkpoint '{file_path}' (epoch {checkpoint['epoch']})")
    else:
        logging.info(f"=> No checkpoint found at '{file_path}'")

# This function can be used to load a checkpoint and restore the model's state.
# Example:
# optimizer = torch.optim.Adam(siamese_net.parameters())
# load_checkpoint(siamese_net, optimizer, "path_to_checkpoint_file.pth.tar")


# Mock training loop with checkpoint saving
def train_siamese_network(model, contrastive_loss, train_dataloader,
                          val_dataloader, optimizer, num_epochs=10,
                          early_stopping_patience=5, save_folder='.'):
    best_val_loss = float('inf')  # Initialize with a high value
    early_stopping_patience_counter = 0
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch No. {epoch}/{num_epochs}")
        
        logging.info(f"[{epoch}/{num_epochs}] Train phase")
        model.train()
        epoch_loss = 0.0  # Track the loss for the current epoch
        
        for i, batch in enumerate(train_dataloader):  
            logging.info(f"[{epoch}/{num_epochs}] [Train phase] batch {i}/{len(train_dataloader)}")
            # input1, input2, target = batch
            input1, input2, _, _, target = batch

            optimizer.zero_grad()
            output1, output2 = model(input1, input2)
            loss = contrastive_loss(output1, output2, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        average_epoch_loss = epoch_loss / len(train_dataloader)
        
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_epoch_loss:.4f}")
        
        
        # Validation loop
        logging.info(f"[{epoch}/{num_epochs}] Eval phase")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # No gradients required for validation
            for i, batch in enumerate(val_dataloader):
                logging.info(f"[{epoch}/{num_epochs}] [Eval phase] batch {i}/{len(val_dataloader)}")
                # input1, input2, target = batch
                input1, input2, _, _, target = batch
                output1, output2 = model(input1, input2)
                loss = contrastive_loss(output1, output2, target)
                val_loss += loss.item()
                
        average_val_loss = val_loss / len(val_dataloader)
        
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {average_val_loss:.4f}")
        
        # Save checkpoint periodically
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            logging.info(f"New best loss {best_val_loss:.4f} achieved! Saving checkpoint.")
            checkpoint_savepath = os.path.join(save_folder, f"checkpoint_{epoch}.pth")
            save_checkpoint(epoch, model, optimizer, checkpoint_savepath)
            logging.info(f"Checkpoint saved to: {checkpoint_savepath}")
            early_stopping_patience_counter = 0
        else:
            early_stopping_patience_counter += 1
        
        if early_stopping_patience_counter >= early_stopping_patience:
            logging.info(f"[{epoch}/{num_epochs}] Early stopping achieved...")
            break
            
    return model
            

def obtain_embeddings(model, dataloader):
    """Obtain embeddings from the model for given dataloader"""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:  
            input1, input2, label1, label2, _ = batch
            # input1, input2, target = batch
            output1, output2 = model(input1, input2)
            embeddings.append(output1.cpu().numpy())
            embeddings.append(output2.cpu().numpy())
            labels.append(label1.cpu().numpy())
            labels.append(label2.cpu().numpy())
            # labels.extend(target.cpu().numpy())
    
    embeddings = np.asarray(flat_list_of_lists(embeddings))
    labels = np.asarray(flat_list_of_lists(labels))
    
    return embeddings, labels

def plot_embeddings_umap(embeddings, labels, unique_markers, savepath=None):
    """Reduce dimensionality of embeddings using UMAP and plot them"""
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)
    labels_flatten = labels.reshape(-1,)
    labels_unique = np.unique(labels_flatten)
    legend = [unique_markers[l] for l in labels_unique]
    
    for l in labels_unique:
        indx = np.where(labels_flatten == l)[0]
        plt.scatter(embeddings_2d[indx, 0], embeddings_2d[indx, 1], cmap='Set1', s=5)
    plt.legend(legend)
    plt.title('UMAP projection of the Embeddings', fontsize=16)
    
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)

# Mock code for testing and visualization (assuming you have a test dataloader 'test_dataloader')
"""
# Load the best checkpoint
optimizer = torch.optim.Adam(siamese_net.parameters())  # re-initialize the optimizer
load_checkpoint(siamese_net, optimizer, "best_val_checkpoint.pth.tar")

# Obtain embeddings for the test set
embeddings, labels = obtain_embeddings(siamese_net, test_dataloader)

# Visualize embeddings using UMAP
plot_embeddings_umap(embeddings, labels)
"""

# Note: The above is mock code and assumes you have a test_dataloader. Uncomment and adjust paths/filenames as necessary.


def get_shape_npy(filename):
    with open(filename, 'rb') as f:
        # Read the header of the npy file
        version, header_len = np.lib.format.read_magic(f)
        
        shape, _, _ = np.lib.format.read_array_header_2_0(f, header_len)
    return shape

# Test the function with a sample .npy file (uncomment the lines below)
# shape = get_shape_npy("path_to_npy_file.npy")
# shape

def flat_list_of_lists(l):
    return [item for sublist in l for item in sublist]

def init_logging(path):
    """Init logging.
    Writes to log file and console.
    Args:
        path (string): Path to log file
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(path),
                            logging.StreamHandler()
                        ])
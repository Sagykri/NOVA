import torch
import numpy as np
import logging
class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_path, labels_path, indexes=None):
        self.embeddings = np.load(embeddings_path)
        self.embeddings = self.embeddings.reshape(len(self.embeddings), -1)
        self.labels = np.load(labels_path)
        self.indexes = indexes if indexes is not None else np.arange(len(self.labels))
        self.embeddings = self.embeddings[indexes]
        self.labels = self.labels[indexes]
        
        self.length = len(self.indexes)
        self.unique_markers = np.unique(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Get an embedding and its label
        embedding1, label1 = self.embeddings[index], self.labels[index]

        # Choose another random index
        # options = np.arange(self.length)
        # options = options[options!=index]
        index2 = np.random.choice(len(self.labels))
        
        while index2 == index:
            index2 = np.random.choice(len(self.labels))
        
        embedding2, label2 = self.embeddings[index2], self.labels[index2]
        target = torch.tensor([float(~~(label1 == label2))])

        label1_onehot = label1[:, None] == self.unique_markers
        label2_onehot = label2[:, None] == self.unique_markers
        
        label1_onehot = label1_onehot.argmax(1)
        label2_onehot = label2_onehot.argmax(1)
        
        return torch.tensor(embedding1, dtype=torch.float32),\
                    torch.tensor(embedding2, dtype=torch.float32),\
                    torch.tensor(label1_onehot), torch.tensor(label2_onehot), target

# You can now initialize the SiameseDataset with the paths to your .npy files:
# siamese_dataset = SiameseDataset("path_to_embeddings.npy", "path_to_labels.npy")


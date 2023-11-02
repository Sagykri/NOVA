import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import torch
from torch import nn

from sandbox.siamese_net.embedding_net import EmbeddingNet

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # We use the same embedding network for both inputs
        self.embedding_net = EmbeddingNet()

    def forward_one(self, x):
        # Forward pass for one input
        return self.embedding_net(x)

    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

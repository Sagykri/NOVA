import torch
import torch.nn as nn
import torch.nn.functional as F

# class EmbeddingNet(nn.Module):
#     def __init__(self):
#         super(EmbeddingNet, self).__init__()
        
#         # Define the architecture of the network
#         self.fc1 = nn.Linear(9216, 4096) #230400
#         self.fc2 = nn.Linear(4096, 1024)
#         self.fc3 = nn.Linear(1024, 256)
#         self.fc4 = nn.Linear(256, 64)
        
#     def forward(self, x):
#         # Forward pass through the network
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)  # We do not apply activation on the last layer to get raw embeddings
#         return x


# class EmbeddingNet(nn.Module):
#     def __init__(self, output_size=128):
#         super(EmbeddingNet, self).__init__()
        
#         # Define a smaller processing unit for each segment
#         self.segment_processor = nn.Sequential(
#             nn.Linear(9216, 256),  # 230400/25 = 9216
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         # Define the main architecture that processes the concatenated output
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256 * 25, 2048),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(2048, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, output_size)
#         )

#     def forward(self, x):
#         # Process each segment separately
#         segments = [self.segment_processor(x[:, i*9216:(i+1)*9216]) for i in range(25)]
#         # Concatenate the processed segments
#         concatenated = torch.cat(segments, dim=1)
#         # Pass through the main architecture
#         return self.fc_layers(concatenated)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
        
    def forward(self, x):
        return x + self.block(x)

class EmbeddingNet(nn.Module):
    def __init__(self, output_size=128):
        super(EmbeddingNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(230400, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            
            ResidualBlock(4096, 4096),
            ResidualBlock(4096, 4096),
            
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            ResidualBlock(1024, 1024),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            ResidualBlock(256, 256),
            
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.model(x)

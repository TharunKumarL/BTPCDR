import torch
from torch_geometric.data import Data

# Load the graph dataset
graphs = torch.load('../data/processed_graphs.pt')

# Check the first graph structure
print(graphs[0])  # This prints the first graph to check its attributes

import torch.nn as nn
import torch_geometric.nn as pyg_nn

class UGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UGCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)  # Aggregate node embeddings to get a single feature vector

# Define model
input_dim = graphs[0].x.shape[1]  # Assuming node features exist
model = UGCN(input_dim=input_dim, hidden_dim=128, output_dim=100)

# Extract features for each drug graph
drug_features = []
for graph in graphs:
    with torch.no_grad():
        feature_vector = model(graph)
        drug_features.append(feature_vector)

# Convert to tensor
drug_features = torch.stack(drug_features)
torch.save(drug_features, '../preprocesseddata/drug_features.pt')

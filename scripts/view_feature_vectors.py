import torch

# Load the feature vectors
drug_features = torch.load('../preprocesseddata/drug_features.pt')

# Print the shape and a few samples
print(f"Feature vectors shape: {drug_features.shape}")  # Should be (223, 100)
print("Sample feature vectors:")
print(drug_features[:5])  # Print the first 5 feature vectors
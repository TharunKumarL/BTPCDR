import torch
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops

def smiles_to_graph(smiles, pubchem_cid):
    """Convert SMILES to graph representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid SMILES

    # Node Features: (75-dim vector per atom)
    X = []
    for atom in mol.GetAtoms():
        atom_features = [
            atom.GetAtomicNum(),  
            atom.GetDegree(),  
            atom.GetHybridization(),  
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge()
        ]
        X.append(atom_features)
    
    X = np.array(X, dtype=np.float32)

    N = mol.GetNumAtoms()
    A = np.zeros((N, N))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = 1
        A[j, i] = 1  # Undirected graph
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # PyG Data object
    return Data(x=X_tensor, edge_index=edge_index, pubchem_id=pubchem_cid)

# Read SMILES file
graph_data_list = []
with open("../data/223drugs_pubchem_smiles.txt", "r") as f:
    for line in f:
        pubchem_cid, smiles = line.strip().split("\t")
        graph_data = smiles_to_graph(smiles, pubchem_cid)
        if graph_data:
            graph_data_list.append(graph_data)
# print(graph_data_list[0])
# data = graph_data_list[0]  # First drug graph
# print("Node Features (x):\n", data.x)
# print("Edge Index:\n", data.edge_index)
# print("PubChem ID:\n", data.pubchem_id)


# Save processed graphs
torch.save(graph_data_list, "../data/processed_graphs.pt")

print(f"Processed {len(graph_data_list)} drug graphs and saved to 'processed_graphs.pt'")

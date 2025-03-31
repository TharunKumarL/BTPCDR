import os
import deepchem as dc
from rdkit import Chem
import hickle as hkl

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
drug_smiles_file = os.path.join(script_dir, '../data/223drugs_pubchem_smiles.txt')
save_dir = os.path.join(script_dir, '../data/GDSC/drug_graph_feat')

# Read SMILES data
pubchemid2smile = {line.split('\t')[0]: line.split('\t')[1].strip() for line in open(drug_smiles_file).readlines()}

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Initialize featurizer
featurizer = dc.feat.graph_features.ConvMolFeaturizer()

# Process molecules
for pubchem_id, smile in pubchemid2smile.items():
    print(f"Processing: {pubchem_id}")

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Warning: Invalid SMILES for {pubchem_id}")
        continue

    # Featurize molecule
    mol_object = featurizer.featurize([mol])

    if len(mol_object) == 0:
        print(f"Error featurizing {pubchem_id}")
        continue

    # Extract features
    features = mol_object[0].atom_features
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list

    # Save data
    hkl.dump([features, adj_list, degree_list], os.path.join(save_dir, f"{pubchem_id}.hkl"))

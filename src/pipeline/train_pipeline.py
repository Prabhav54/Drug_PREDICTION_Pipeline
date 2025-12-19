import sys
import os
import pandas as pd
from src.utils import load_object
from rdkit import Chem
from rdkit.Chem import Descriptors

class PredictPipeline:
    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        # Same constants as build_dataset.py
        self.pdb_features = {
            'CA2':  {'Pocket_Volume': 450.2, 'Electrostatics': -12.5},
            'CA9':  {'Pocket_Volume': 410.5, 'Electrostatics': -8.2},
            'CA12': {'Pocket_Volume': 430.1, 'Electrostatics': -10.1}
        }

    def predict(self, smiles):
        results = {}
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return "Invalid SMILES"

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        for iso in ['CA2', 'CA9', 'CA12']:
            # Create a dataframe for the model input
            data = {
                'isoform_target': [iso],
                'Mol_Weight': [mw],
                'LogP': [logp],
                'Pocket_Volume': [self.pdb_features[iso]['Pocket_Volume']],
                'Electrostatics': [self.pdb_features[iso]['Electrostatics']]
            }
            df = pd.DataFrame(data)
            scaled_data = self.preprocessor.transform(df)
            pred = self.model.predict(scaled_data)[0]
            results[iso] = pred

        # Selectivity: CA9 (Target) - CA2 (Off-Target)
        si = results['CA9'] - results['CA2']
        
        return {
            "CA9_Affinity": round(results['CA9'], 2),
            "CA12_Affinity": round(results['CA12'], 2),
            "CA2_Affinity": round(results['CA2'], 2),
            "Selectivity_Index": round(si, 2),
            "Conclusion": "Selective" if si > 1.0 else "Non-Selective/Toxic"
        }
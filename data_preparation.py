import os
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client

# Scientific Constants (Hardcoded for speed, as discussed)
PROTEIN_FEATURES = {
    'CA2':  {'Pocket_Volume': 450.2, 'Electrostatics': -12.5}, # Off-Target
    'CA9':  {'Pocket_Volume': 410.5, 'Electrostatics': -8.2},  # Target
    'CA12': {'Pocket_Volume': 430.1, 'Electrostatics': -10.1}  # Target
}

TARGETS = {
    "CA2": "CHEMBL205",
    "CA9": "CHEMBL3658",
    "CA12": "CHEMBL4698"
}

def fetch_data():
    print("--- 1. Fetching Data from ChEMBL (API) ---")
    all_data = []
    activity = new_client.activity

    for isoform, chembl_id in TARGETS.items():
        print(f"   Fetching {isoform}...")
        res = activity.filter(target_chembl_id=chembl_id)\
                      .filter(standard_type="IC50")\
                      .filter(standard_relation="=")\
                      .only(['molecule_chembl_id', 'standard_value', 'canonical_smiles'])
        
        for item in res:
            if item.get('canonical_smiles') and item.get('standard_value'):
                try:
                    ic50 = float(item['standard_value'])
                    if 0 < ic50 < 100000: # Filter outliers
                        pic50 = -np.log10(ic50 * 1e-9)
                        
                        all_data.append({
                            "interaction_id": f"{item['molecule_chembl_id']}_{isoform}",
                            "ligand_id": item['molecule_chembl_id'],
                            "isoform_target": isoform,
                            "SMILES": item['canonical_smiles'],
                            # Inject Protein Features
                            "Pocket_Volume": PROTEIN_FEATURES[isoform]['Pocket_Volume'],
                            "Electrostatics": PROTEIN_FEATURES[isoform]['Electrostatics'],
                            "pIC50": round(pic50, 3)
                        })
                except: continue

    df = pd.DataFrame(all_data)
    os.makedirs('artifacts', exist_ok=True)
    df.to_csv('artifacts/data.csv', index=False)
    print(f"--- Success! Saved {len(df)} rows to artifacts/data.csv ---")

if __name__ == "__main__":
    fetch_data()
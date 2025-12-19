import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_transformer_object(self):
        # Numerical features: MW, LogP, Pocket Volume, Electrostatics
        num_cols = ["Mol_Weight", "LogP", "Pocket_Volume", "Electrostatics"]
        # Categorical: Isoform (Target)
        cat_cols = ["isoform_target"]

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])

        return ColumnTransformer([
            ("num_pipeline", num_pipeline, num_cols),
            ("cat_pipeline", cat_pipeline, cat_cols)
        ])

    def generate_rdkit_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return pd.Series([Descriptors.MolWt(mol), Descriptors.MolLogP(mol)])
            return pd.Series([np.nan, np.nan])
        except:
            return pd.Series([np.nan, np.nan])

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print("--- Generating RDKit Features (MW, LogP) ---")
            train_df[["Mol_Weight", "LogP"]] = train_df["SMILES"].apply(self.generate_rdkit_features)
            test_df[["Mol_Weight", "LogP"]] = test_df["SMILES"].apply(self.generate_rdkit_features)
            
            train_df.dropna(subset=["Mol_Weight"], inplace=True)
            test_df.dropna(subset=["Mol_Weight"], inplace=True)

            preprocessing_obj = self.get_transformer_object()

            target_col = "pIC50"
            drop_cols = ["interaction_id", "ligand_id", "SMILES", target_col]

            input_train = train_df.drop(columns=drop_cols, axis=1)
            target_train = train_df[target_col]

            input_test = test_df.drop(columns=drop_cols, axis=1)
            target_test = test_df[target_col]

            input_train_arr = preprocessing_obj.fit_transform(input_train)
            input_test_arr = preprocessing_obj.transform(input_test)

            train_arr = np.c_[input_train_arr, np.array(target_train)]
            test_arr = np.c_[input_test_arr, np.array(target_test)]

            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)
            print("--- Transformation Complete: Preprocessor Saved ---")

            return train_arr, test_arr

        except Exception as e:
            raise Exception(f"Error in Transformation: {e}")
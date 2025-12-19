import os
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model = XGBRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X_train, y_train)

            predicted = model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            
            save_object(self.config.trained_model_file_path, model)
            print(f"--- Model Training Complete. R2 Score: {r2} ---")
            return r2

        except Exception as e:
            raise Exception(f"Error in Training: {e}")
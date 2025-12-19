from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 1. Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Transformation
    transform = DataTransformation()
    train_arr, test_arr = transform.initiate_data_transformation(train_path, test_path)

    # 3. Training
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)
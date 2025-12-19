# Automated ML Pipeline for Carbonic Anhydrase Drug Discovery

An end-to-end Machine Learning pipeline to predict isoform-selective inhibitors for Carbonic Anhydrase IX (Cancer Target) vs CA II (Off-Target).

## Project Structure
- `src/components`: Data Ingestion, Transformation, Model Training
- `src/pipeline`: Training and Prediction Pipelines
- `artifacts`: Stores generated datasets and models
- `application.py`: Flask Web App endpoint

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
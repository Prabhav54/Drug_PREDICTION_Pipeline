from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form.get('smiles')
    pipeline = PredictPipeline()
    results = pipeline.predict(smiles)
    return render_template('index.html', results=results, smiles=smiles)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
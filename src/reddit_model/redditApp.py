from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying Reddit comments'}

class request_body(BaseModel):
    reddit_comment : str

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    X = [data.reddit_comment]
    predictions = model_pipeline.predict_proba(X)
    return {'Predictions': predictions}
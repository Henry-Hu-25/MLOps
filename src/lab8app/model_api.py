from fastapi import FastAPI
import mlflow
import uvicorn
from pydantic import BaseModel
import pandas as pd


mlflow.set_tracking_uri("sqlite:///lab6.db")
mlflow.set_experiment("Metaflow_Scoring")

app = FastAPI(
   title = "Car Price Prediction",
   description = "Predict the price of a used car based on its features",
   version = "1.0.0"
)


@app.get("/")
def main():
    return {"message": "Welcome to the Car Price Prediction API"}

class Car(BaseModel):
    odometer: float
    age: int

model = None

@app.on_event("startup")
def load_model():
    model_uri = f"runs:/164d2aa8606940fca28881ee18cb2798/model"
    global model
    model = mlflow.sklearn.load_model(model_uri)
load_model()
        

@app.post("/predict")
def predict(car: Car):
    # Convert the car object to a DataFrame
    car_dict = car.dict()
    car_df = pd.DataFrame([car_dict])
    
    # Make prediction
    prediction = model.predict(car_df)
    
    return {"prediction": float(prediction[0])}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
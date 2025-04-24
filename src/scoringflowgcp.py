from metaflow import FlowSpec, step, Parameter, conda_base, conda, kubernetes, resources, timeout, retry, catch
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
from dvc_preprocessing import preprocess_data

mlflow.set_tracking_uri("sqlite:///lab6.db")
mlflow.set_experiment("Metaflow_Scoring")

@conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.3.0'}, python='3.9.16')
class ScoreFlow(FlowSpec):

    run_id = Parameter(
        'run_id',
        help='MLflow run ID containing the model to load',
        required=True
    )

    @step
    def start(self):
        self.next(self.load_data)

    @step
    @retry(times=3)
    def load_data(self):
        _, self.X_test, _, self.y_test = preprocess_data()
        self.next(self.load_model)

    @step
    @timeout(minutes=10)
    @catch(var='model_error')
    def load_model(self):
        if hasattr(self, 'model_error'):
            print(f"Caught an error loading model: {self.model_error}")
            self.next(self.end)
            return
            
        model_uri = f"runs:/{self.run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.score)

    @step
    @kubernetes
    def score(self):
        preds = self.model.predict(self.X_test)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, preds))

        with mlflow.start_run(run_name="scoring", nested=True):
            mlflow.log_metric("RMSE", self.rmse)

        print(f"Scoring results → RMSE: {self.rmse:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print("ScoringFlow complete.")

if __name__ == "__main__":
    ScoreFlow()

from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from dvc_preprocessing import preprocess_data

mlflow.set_tracking_uri("sqlite:///lab6.db")
mlflow.set_experiment("Metaflow_Scoring")

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
    def load_data(self):
        _, self.X_test, _, self.y_test = preprocess_data()
        self.next(self.load_model)

    @step
    def load_model(self):
        model_uri = f"runs:/{self.run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.score)

    @step
    def score(self):
        preds = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, preds.round())

        with mlflow.start_run(run_name="scoring", nested=True):
            mlflow.log_metric("Accuracy", self.accuracy)

        print(f"Scoring results → Accuracy: {self.accuracy:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print("ScoringFlow complete.")

if __name__ == "__main__":
    ScoreFlow()

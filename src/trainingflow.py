from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, Trials
from hyperopt import hp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from dvc_preprocessing import preprocess_data

mlflow.set_tracking_uri("sqlite:///lab6.db")
mlflow.set_experiment("Metaflow_TrainFlow")

def objective(params, X_train, y_train):
    p = params.copy()
    regressor_type = p.pop('type')
    if regressor_type == 'dt':
        model = DecisionTreeRegressor(**p)
    elif regressor_type == "rf":
        model = RandomForestRegressor(**p)
    else:
        model = LinearRegression(**p)

    accuracy = cross_val_score(
        model, X_train, y_train, scoring='r2', cv=5
    ).mean()

    with mlflow.start_run(nested=True):
        mlflow.set_tag("Model", regressor_type)
        mlflow.log_params(p)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}

searched_space = hp.choice('regressor_type', [
    { 'type': 'dt',
      'criterion': hp.choice('dtree_criterion', ['squared_error', 'absolute_error']),
      'max_depth': hp.choice('dtree_max_depth', [None, hp.randint('dtree_max_depth_int', 1, 10)]),
      'min_samples_split': hp.randint('dtree_min_samples_split', 2, 10)
    },
    { 'type': 'rf',
      'n_estimators': hp.randint('rf_n_estimators', 20, 100),
      'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', None]),
      'criterion': hp.choice('rf_criterion', ['squared_error', 'absolute_error'])
    },
    { 'type': 'linear',
      'fit_intercept': True,
      'positive': False
    }
])

class TrainFlow(FlowSpec):
    runs = Parameter("runs", default=5, type=int, help="Number of Hyperopt trials")

    @step
    def start(self):
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data()
        self.next(self.train)

    @step
    def train(self):
        trials = Trials()
        with mlflow.start_run(run_name="hyperopt_parent"):
            best = fmin(
                fn=lambda params: objective(params, self.X_train, self.y_train),
                space=searched_space,
                algo=tpe.suggest,
                max_evals=self.runs,
                trials=trials
            )

        best_idx = int(np.argmin([t['result']['loss'] for t in trials.trials]))
        best_model = trials.trials[best_idx]['result']['model']
        best_model.fit(self.X_train, self.y_train)

        with mlflow.start_run(run_name="best_params"):
            mlflow.log_params(best)
            mlflow.sklearn.log_model(best_model, artifact_path="model")
        
        run_id = mlflow.last_active_run().info.run_id
        mlflow.register_model(f"runs:///{run_id}/artifacts/model", "best_model")

        self.best_params = best
        self.next(self.end)

    @step
    def end(self):
        print(f"Best params: {self.best_params}")
        print("Flow complete!")

if __name__ == "__main__":
    TrainFlow()



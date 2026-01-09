import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

class MLflowExperimentTracker:
    def __init__(self, experiment_name="fraud-detection", tracking_uri=None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def train_with_tracking(self, X_train, y_train, X_val, y_val, params):
        """Train model with MLflow tracking"""

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)

            # Train model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=False
            )

            # Make predictions
            y_pred = model.predict(dval)
            y_pred_binary = (y_pred > 0.5).astype(int)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred_binary),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1_score': f1_score(y_val, y_pred_binary),
                'roc_auc': roc_auc_score(y_val, y_pred)
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.xgboost.log_model(model, "model")

            # Log additional info
            mlflow.set_tags({
                'training_date': datetime.now().isoformat(),
                'model_type': 'xgboost',
                'dataset_size': len(X_train)
            })

            print(f"Run ID: {run.info.run_id}")
            print(f"Metrics: {metrics}")

            return model, run.info.run_id

    def register_best_model(self, run_id, model_name="fraud-detection-model"):
        """Register model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        return result

    def promote_model_to_production(self, model_name, version):
        """Promote model version to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        print(f"Model {model_name} version {version} promoted to Production")

    def get_production_model(self, model_name):
        """Get current production model"""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])

        if versions:
            return versions[0]
        return None

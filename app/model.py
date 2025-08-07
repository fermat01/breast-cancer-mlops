import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from utils import FEATURE_ORDER  


# Save model inside the same directory as this script (e.g. app/rf_model.joblib)
MODEL_FILENAME = os.path.join(os.path.dirname(__file__), "rf_model.joblib")


def train_and_log_model():
    mlflow.set_experiment("Breast_Cancer_Classification")

    data = load_breast_cancer(as_frame=True)
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest_model")

        joblib.dump(model, MODEL_FILENAME)
        print(f"Model trained and saved to {MODEL_FILENAME} with accuracy: {acc:.4f}")

        return model


def load_model():
    return joblib.load(MODEL_FILENAME)


def predict(model, features):
    return model.predict(features)


if __name__ == "__main__":
    train_and_log_model()

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import dagshub
import logging
import sqlite3

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Initialize DagsHub
dagshub.init(repo_owner='rambabar', repo_name='mlops_streamlit_dagshub', mlflow=True)
remote_server_uri="https://dagshub.com/rambabar/mlops_streamlit_dagshub.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
# export MLFLOW_TRACKING_URI=https://dagshub.com/rambabar/mlops_streamlit_dagshub.mlflow
# export MLFLOW_TRACKING_USERNAME=rambabar
# export MLFLOW_TRACKING_PASSWORD=2e01294e4d01ba2f178010aec555e3e26692cbf7

def train_model(retrain=False):
    if retrain or not os.path.exists("model"):
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        csv_url = (
            "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
            )
        try:
            data = pd.read_csv(csv_url, sep=";")
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )
        train, test = train_test_split(data)
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)

            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    lr, "model", registered_model_name="ElasticnetWineModel"
                )
            else:
                mlflow.sklearn.log_model(lr, "model")

def load_latest_model():
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    latest_run_id = runs.iloc[0].run_id
    latest_model_uri = f"runs:/{latest_run_id}/model"
    return mlflow.sklearn.load_model(latest_model_uri)

def predict_with_latest_model(data):
    model = load_latest_model()
    return model.predict(data)

def init_db():
    conn = sqlite3.connect('user_inputs.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature1 REAL,
        feature2 REAL,
        feature3 REAL,
        feature4 REAL,
        feature5 REAL,
        feature6 REAL,
        feature7 REAL,
        feature8 REAL,
        feature9 REAL,
        feature10 REAL,
        feature11 REAL,
        prediction REAL
    )
    ''')
    conn.commit()
    conn.close()

def save_user_input_with_prediction(data, prediction):
    conn = sqlite3.connect('user_inputs.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO user_data (feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, prediction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (*data, prediction))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    retrain = False  # Set to True to retrain model on random synthetic data
    if retrain:
        train_model(retrain)
    else:
        pass

    # Initialize database and table
    init_db()

    # Example of saving user input to SQLite database
    user_input = np.random.rand(11).tolist()  # Replace with actual user input

    # Predicting with the latest model
    user_input_np = np.array(user_input).reshape(1, -1)  # Reshape for prediction
    prediction = predict_with_latest_model(user_input_np)[0]  # Get the prediction
    print("Prediction:", prediction)

    # Save the user input and prediction in the database
    save_user_input_with_prediction(user_input, prediction)

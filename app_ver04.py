import os, time
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
import streamlit as st

import os
from google.cloud import storage

def upload_experiment_folder(bucket_name, source_directory, destination_folder):
    """Uploads the experiment folder to Google Cloud Storage if it does not exist.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        source_directory (str): The path to the source directory.
        destination_folder (str): The destination folder in the bucket.
    """
    # Create a Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Check if the destination folder exists in the bucket
    blobs = list(bucket.list_blobs(prefix=destination_folder))
    if not blobs:
        # Upload the experiment folder to the bucket
        for root, _, files in os.walk(source_directory):
            for file in files:
                blob_name = os.path.relpath(os.path.join(root, file), source_directory)
                blob = bucket.blob(os.path.join(destination_folder, blob_name))
                blob.upload_from_filename(os.path.join(root, file))
                print(f"Uploaded {blob_name} to {bucket_name}/{destination_folder}")

# Example usage
bucket_name = "ml_model_data"
source_directory = "mlruns"
destination_folder = "mlruns"
upload_experiment_folder(bucket_name, source_directory, destination_folder)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Initialize DagsHub
# dagshub.init(repo_owner='rambabar', repo_name='mlflow', mlflow=True)
# remote_server_uri="https://dagshub.com/rambabar/mlflow.mlflow"
# mlflow.set_tracking_uri(remote_server_uri)

def train_model():
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    local_csv_path = os.path.join(os.getcwd(), "winequality-red.csv")
    if os.path.exists(local_csv_path):
        data = pd.read_csv(local_csv_path, sep=";")
    else:
        csv_url = (
            "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
            )
        try:
            data = pd.read_csv(csv_url, sep=";")
            data.to_csv(local_csv_path, index=False)
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )
    print(data)
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        print("Training model")
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


def retrain_model():
    local_csv_path = os.path.join(os.getcwd(), "winequality-red.csv")
    if os.path.exists(local_csv_path):
        data = pd.read_csv(local_csv_path, sep=";")
    else:
        csv_url = (
            "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
            )
        try:
            data = pd.read_csv(csv_url, sep=";")
            data.to_csv(local_csv_path)
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )
    # Load data from the database
    conn = sqlite3.connect('user_inputs.db')
    df = pd.read_sql_query("SELECT * FROM user_data", conn)
    #conn.execute("DELETE FROM user_data").commit()
    conn.close()
    print(df)
    df.drop(columns=['id'], inplace=True)

    X = df.drop(columns=['prediction']).values
    y = df['prediction'].values
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define alpha and l1_ratio
    alpha = 0.5  # default value
    l1_ratio = 0.5  # default value

    # Train the model
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    # Make predictions
    predicted_qualities = lr.predict(test_x)

    # Evaluate metrics
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    # Print and log metrics
    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.start_run()
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

    mlflow.end_run()

    st.success("Model retrained successfully!")

def main():
    if not os.path.exists("mlruns"):
        train_model()

    st.title('ElasticNet Wine Quality Predictor')
    st.header('Input Features')

    feature_names = [
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
        "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density",
        "pH", "Sulphates", "Alcohol"
    ]

    # Divide the page into multiple rows of columns
    rows = 3  # Number of rows
    cols_per_row = 4  # Number of columns per row
    total_inputs = 11

    # Create input fields for each feature
    features = []
    for i in range(rows):
        cols = st.columns(cols_per_row)  # Create columns for each row
        for j, col in enumerate(cols):
            idx = i * cols_per_row + j + 1  # Calculate index for the input feature
            if idx <= total_inputs:
                feature_idx = idx - 1  # Adjusting for 0-based indexing
                features.append(col.number_input(f'{feature_names[feature_idx]}', value=np.random.rand(), step=0.01))

    import matplotlib.pyplot as plt

    # features = []
    # for feature_name in feature_names:
    #     features.append(st.number_input(feature_name, value=np.random.rand(), step=0.01))

    # Organize buttons in a single row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button('Predict', key=1):
            # Convert the features to a numpy array
            user_input_np = np.array(features).reshape(1, -1)

            # Load the latest model and make prediction
            prediction = predict_with_latest_model(user_input_np)[0]

            st.success(f'Predicted Quality: {prediction}')
            print(f'Predicted Quality: {prediction}')

            # Save the user input and prediction in the database
            save_user_input_with_prediction(features, prediction)
            
            # Prepare the data for plotting
            data_point = user_input_np[0]
            fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the size here (width, height)
            ax.plot(feature_names, data_point, marker='o', linestyle='-')
            plt.xticks(rotation=90)
            st.pyplot(fig)

    with col2:
        if st.button('Retrain Model'):
            retrain_model()

    with col3:
        if st.button('Reset Model'):
            folder_path = "mlruns"
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                [os.remove(os.path.join(root, file)) for root, dirs, files in os.walk(folder_path) for file in files] or [os.rmdir(os.path.join(root, dir)) for root, dirs, files in os.walk(folder_path, topdown=False) for dir in dirs] or os.rmdir(folder_path)
                print(f"Folder '{folder_path}' has been removed successfully.")
            else:
                print(f"Folder '{folder_path}' does not exist or is not a directory.")
            mlflow.set_tracking_uri("file:///mlruns")
            train_model()
            st.success("Model reset successfully!")
    
    with col4:
        if st.button('Remove Old Inputs'):
            conn = sqlite3.connect('user_inputs.db')
            conn.execute("DELETE FROM user_data").commit()
            conn.close()
        
if __name__ == "__main__":
    main()

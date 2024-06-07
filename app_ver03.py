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
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='auto')
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Initialize DagsHub
# dagshub.init(repo_owner='rambabar', repo_name='mlops_streamlit_dagshub', mlflow=True)
remote_server_uri="https://dagshub.com/rambabar/mlops_streamlit_dagshub.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

dagshub.init()
def train_model():
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
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Load data from the database
    conn = sqlite3.connect('user_inputs.db')
    df = pd.read_sql_query("SELECT * FROM user_data", conn)
    #conn.execute("DELETE FROM user_data").commit()
    conn.close()

    if df.shape[0]==0:
        st.success("No user inputs in database")
    else:
        df.drop(columns=['id'], inplace=True)
        data.columns = df.columns
        df = pd.concat([df, data], ignore_index=True)
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
        st.link_button("Training experiment link", remote_server_uri)

def main():
    if not os.path.exists("mlruns"):
        train_model()

    st.title('ElasticNet Wine Quality Predictor')
    st.link_button("Github Code", "https://github.com/rambabar/mlops_streamlit_dagshub")
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


    # Organize buttons in a single row
    col1, col2, col3 = st.columns([1, 1, 1])

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

            conn = sqlite3.connect('user_inputs.db')
            df = pd.read_sql_query("SELECT * FROM user_data", conn)
            #conn.execute("DELETE FROM user_data").commit()
            conn.close()
            df = df.tail(10)
            st.write("Last input analysis (upto 10)")
            df.drop(columns=['id'], inplace=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(kind='bar', ax=ax)
            plt.title('User Input and Predicted Quality')
            plt.xlabel('Samples')
            plt.ylabel('Values')
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

    with col2:
        if st.button('Retrain Model'):
            retrain_model()

    with col3:
        if st.button('Remove Old Inputs'):
            conn = sqlite3.connect('user_inputs.db')
            conn.execute("DELETE FROM user_data")
            conn.commit()  # Call commit() on the connection object
            conn.close()
            st.success(f'Removed all Inputs.')

if __name__ == "__main__":
    main()

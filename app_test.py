import os
import warnings
import sys
import sqlite3
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import dagshub
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Initialize DAGsHub
dagshub.init(repo_owner='rambabar', repo_name='mlops_streamlit_dagshub', mlflow=True)
remote_server_uri = "https://dagshub.com/rambabar/mlops_streamlit_dagshub.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

# Database connection
conn = sqlite3.connect('wine_quality.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        fixed_acidity REAL, volatile_acidity REAL, citric_acid REAL,
        residual_sugar REAL, chlorides REAL, free_sulfur_dioxide REAL,
        total_sulfur_dioxide REAL, density REAL, pH REAL, sulphates REAL,
        alcohol REAL, quality REAL, prediction REAL
    )
''')
conn.commit()

# Sample input data
input_data = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Load the model
model_uri = 'models:/ElasticnetWineModel/Production'
try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully!")
except mlflow.exceptions.MlflowException as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Make a prediction
prediction = model.predict(input_df)[0]
print(f'The predicted wine quality is: {prediction}')

# Save prediction to database
input_data['prediction'] = prediction
columns = ', '.join(input_data.keys())
placeholders = ', '.join('?' * len(input_data))
c.execute(f'INSERT INTO predictions ({columns}) VALUES ({placeholders})', tuple(input_data.values()))
conn.commit()
print("Prediction saved to database.")

# Retrain the model with new data
def retrain_model():
    c.execute('SELECT * FROM predictions')
    data = pd.DataFrame(c.fetchall(), columns=[*input_data.keys(), 'quality'])
    X = data.drop(columns=['quality', 'prediction'])
    y = data['quality']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)

    with mlflow.start_run():
        alpha = 0.5
        l1_ratio = 0.5
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

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")

# Uncomment the line below to retrain the model
# retrain_model()

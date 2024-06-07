import streamlit as st
import numpy as np
import sqlite3
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import mlflow.sklearn
import pandas as pd

def predict_with_latest_model(data):
    model = load_latest_model()
    return model.predict(data)

def load_latest_model():
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    latest_run_id = runs.iloc[0].run_id
    latest_model_uri = f"runs:/{latest_run_id}/model"
    return mlflow.sklearn.load_model(latest_model_uri)

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
    # Load data from the database
    conn = sqlite3.connect('user_inputs.db')
    df = pd.read_sql_query("SELECT * FROM user_data", conn)
    conn.close()

    # Prepare features and target
    X = df.drop(columns=['prediction']).values
    y = df['prediction'].values

    # Retrain the model
    alpha = 0.5  # default value
    l1_ratio = 0.5  # default value
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X, y)

    # Log the retrained model
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")

    st.success("Model retrained successfully!")

def main():
    st.title('ElasticNet Wine Quality Predictor')
    

    st.header('Input Features')

    # Create input fields for each feature
    features = []
    for i in range(1, 12):
        features.append(st.number_input(f'Feature {i}', value=np.random.rand(), step=0.01))

    if st.button('Predict'):
        # Convert the features to a numpy array
        user_input_np = np.array(features).reshape(1, -1)

        # Load the latest model and make prediction
        prediction = predict_with_latest_model(user_input_np)[0]

        st.success(f'Predicted Quality: {prediction}')

        # Save the user input and prediction in the database
        save_user_input_with_prediction(features, prediction)

    # Retrain button
    if st.button('Retrain Model'):
        retrain_model()

if __name__ == "__main__":
    main()

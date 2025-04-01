# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    return parser.parse_args()

def main(args):
    '''Read train dataset, train model, save trained model'''
    # Read train data
    train_data = pd.read_csv(Path(args.train_data))
    print(train_data.head())
    print(train_data.columns)
    # Split the data into input(X) and output(y)
    TARGET_COL = 'Rent'
    y_train = train_data[TARGET_COL]
    X_train = train_data.drop(columns=TARGET_COL)

    numeric_features = [col for col in X_train.columns if X_train[col].dtype != 'object']
    categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # Normalize numerical features
            ('cat', OneHotEncoder(drop='first'), categorical_features)  # One-hot encode categorical features
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # log model hyperparameters
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("model", "LinearRegression")

    # Train model with the train set
    pipeline.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = pipeline.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=pipeline, path=args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()
    TARGET_COL = 'Rent'
    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"TARGET_COL: {TARGET_COL}"  # Log the TARGET_COL used
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()


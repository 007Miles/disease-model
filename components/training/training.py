import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

def main(args):
    # Start Logging
    mlflow.start_run()

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Read training data
    train_df = get_dataframe(args.train_data)
    y_train = train_df.pop("Diabetic")
    X_train = train_df.values

    # Read test data
    test_df = get_dataframe(args.test_data)
    y_test = test_df.pop("Diabetic")
    X_test = test_df.values

    # Train the model
    model = train_model(args.reg_rate, X_train, y_train)

    # Evaluate the model
    model_evaluate(model, X_test, y_test)

    # Register and save the model
    register_and_save_model(model, args.registered_model_name, args.model_save_path)

    # Stop MLflow logging
    mlflow.end_run()

# Function to get the CSV file path
def get_csv(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    raise FileNotFoundError("No CSV file found in the provided directory")

def get_dataframe(path):
    """Reads a CSV file and returns a DataFrame."""
    print("Reading data...")
    csv_path = get_csv(path)
    df = pd.read_csv(csv_path)
    return df

def train_model(reg_rate, X_train, y_train):
    """Trains a logistic regression model."""
    print("Training model...")
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
    return model

def model_evaluate(model, X_test, y_test):
    """Evaluates the model and prints accuracy and AUC score. Also plots the ROC curve."""
    # Calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    # Calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print('AUC:', auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal 50% line
    plt.plot(fpr, tpr)  # Plot the FPR and TPR achieved by our model
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def register_and_save_model(model, model_name, model_save_path):
    """Registers the model with MLflow and saves it to a specified path."""
    print("Registering the model via MLFlow...")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=model_name,
        artifact_path=model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(model_save_path, "trained_model"),
    )

def get_args():
    """Parses and returns command line arguments."""
    parser = argparse.ArgumentParser(description="Argument parser for training script")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--reg_rate", required=False, default=0.1, type=float, help="Regularization rate")
    parser.add_argument("--registered_model_name", type=str, help="Registered model name")
    parser.add_argument("--model_save_path", type=str, help="Path to save the model")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)

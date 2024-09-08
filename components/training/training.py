import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score

def main(args):
    # Start Logging
    mlflow.start_run()

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Read training data
    train_df = get_dataframe(args.train_data)
    X_train, y_train = get_features_and_target(train_df)

    # Read test data
    test_df = get_dataframe(args.test_data)
    X_test, y_test = get_features_and_target(test_df)

    # Train the model
    model = train_model(X_train, y_train, args.n_estimators)

    # Evaluate the model
    model_evaluate(model, X_test, y_test, X_train, y_train)

    # Register and save the model
    register_and_save_model(model, args.registered_model_name, args.model_save_path)

    # Stop MLflow logging
    mlflow.end_run()

def get_csv(directory):
    """Helper function to get the CSV file path from a directory."""
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

def get_features_and_target(df):
    """Separates features and target variable from the DataFrame."""
    # Define features and target
    feature_columns = [
        'NDVI',
        'RENDVI',
        'CIRE',
        'PRI',
        'Temperature',
        'Humidity',
        'UV_Level'
    ]
    X = df[feature_columns]
    y = df['Brown_Blight']
    return X, y

def train_model(X_train, y_train, n_estimators):
    """Trains a Random Forest Classifier."""
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced'
    ).fit(X_train, y_train)
    return model

def model_evaluate(model, X_test, y_test, X_train, y_train):
    """Evaluates the model and prints accuracy and classification report."""
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    mlflow.log_metric('accuracy', accuracy)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    cv_mean = cv_scores.mean()
    print(f'10-Fold Cross-Validation Accuracy: {cv_mean * 100:.2f}%')
    mlflow.log_metric('cv_accuracy', cv_mean)

    # Print classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print('AUC:', auc)
    mlflow.log_metric('auc', auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.plot(fpr, tpr, label='Random Forest')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.show()

def register_and_save_model(model, model_name, model_save_path):
    """Registers the model with MLflow and saves it to a specified path."""
    print("Registering the model via MLflow...")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=model_name,
        artifact_path=model_name,
    )

    # Saving the model to a file
    os.makedirs(model_save_path, exist_ok=True)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(model_save_path, "trained_model"),
    )
    print(f"Model saved to: {model_save_path}")

def get_args():
    """Parses and returns command line arguments."""
    parser = argparse.ArgumentParser(description="Argument parser for training script")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int, help="Number of trees in the forest")
    parser.add_argument("--registered_model_name", type=str, help="Registered model name")
    parser.add_argument("--model_save_path", type=str, help="Path to save the model")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)

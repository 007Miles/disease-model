# This script performs the task of loading the disease dataset,
# preprocessing it, and splitting it into train and test datasets.

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

def main():
    """Main function of the script."""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input disease dataset CSV")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.3)
    parser.add_argument("--train_data", type=str, help="Path to save preprocessed train data")
    parser.add_argument("--test_data", type=str, help="Path to save preprocessed test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("Input data path:", args.data)

    # Read the disease dataset CSV
    df = pd.read_csv(args.data)

    # Drop unnecessary columns if any (e.g., IDs or non-feature columns)
    # df.drop(columns=["UnnecessaryColumn"], inplace=True)

    # Log dataset metrics
    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", df.shape[1] - 1)  # Assuming last column is the target

    # Split the data into training and testing datasets
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_train_ratio,
        random_state=42,
        stratify=df['Brown_Blight']  # Ensure the target variable is stratified
    )

    # Save the training data to the specified output path
    os.makedirs(args.train_data, exist_ok=True)
    train_output_path = os.path.join(args.train_data, "train_data.csv")
    train_df.to_csv(train_output_path, index=False)
    
    # Save the testing data to the specified output path
    os.makedirs(args.test_data, exist_ok=True)
    test_output_path = os.path.join(args.test_data, "test_data.csv")
    test_df.to_csv(test_output_path, index=False)

    print(f"Training data saved to: {train_output_path}")
    print(f"Testing data saved to: {test_output_path}")

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()

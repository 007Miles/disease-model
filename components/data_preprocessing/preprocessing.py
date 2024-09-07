# This script performs the simple task of splitting the data into train and test datasets to represent data preprocessing compoenent code script.

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="Path to save preprocessed train data")
    parser.add_argument("--test_data", type=str, help="Path to save preprocessed test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    diabetes_df = pd.read_csv(args.data)
    diabetes_df.drop(columns=["PatientID"], inplace=True)

    mlflow.log_metric("num_samples", diabetes_df.shape[0])
    mlflow.log_metric("num_features", diabetes_df.shape[1] - 1)

    diabetes_train_df, diabetes_test_df = train_test_split(
        diabetes_df,
        test_size=args.test_train_ratio,
    )

     # Save the training data to the specified output path
    train_output_path = os.path.join(args.train_data, "train_data.csv")
    diabetes_train_df.to_csv(train_output_path, index=False)
    
    # Save the testing data to the specified output path
    test_output_path = os.path.join(args.test_data, "test_data.csv")
    diabetes_test_df.to_csv(test_output_path, index=False)

    print(f"Training data saved to: {train_output_path}")
    print(f"Testing data saved to: {test_output_path}")

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
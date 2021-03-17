"""
    This script contains code to downsample training data
    Dependencies :
        ../model_data_prep/model_data_prep.py
        ../model_data_prep/model_data_eda_and_feature_engg.ipynb
        ../model_data_prep/data_cleaning_and_data_split.ipynb
"""

# Load packages
from pathlib import Path
import numpy as np
from sklearn.utils import resample


def downsample_function(file_type: str, downsample_fraction: float = 0.30):
    """
        Function to downsample data
    """
    # Load train dataset
    data_folder = Path.cwd().parents[0].joinpath('data', 'processed_data')
    x_train = np.load(data_folder.joinpath(file_type + '_encode.npy'))
    y_train = np.load(data_folder.joinpath('y_train.npy'))

    # Downsample data with class 0
    x_train_0 = resample(x_train[y_train == 0], replace=False,
                         n_samples=int(len(x_train[y_train == 0])*downsample_fraction),
                         random_state=10)

    x_train_downsample = np.vstack((x_train_0, x_train[y_train == 1]))
    y_train_downsample = np.hstack((np.zeros(len(x_train_0), dtype=int), np.ones(len(x_train[y_train == 1]),
                                                                                 dtype=int)))

    # Save downsampled data
    np.save(data_folder.joinpath(file_type + '_downsampled.npy'), x_train_downsample)
    np.save(data_folder.joinpath('y_train_downsampled.npy'), y_train_downsample)

    print('Data downsampled and saved successfully')


if __name__ == "__main__":
    downsample_function("x_train_num")  # Use x_train_onehot to use x_train with one hot encoded categorical features

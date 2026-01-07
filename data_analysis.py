import numpy as np
from scipy.stats import skew, kurtosis

DATASET_PATH = "__data/pisa_data_2025_12_02.h5"


def analyze_distribution(data):

    # mean and std of each features
    mean = np.mean(data, axis=0)[0]
    std = np.std(data, axis=0)[0]
    # min and max of distribution
    min_val = np.min(data, axis=0)[0]
    max_val = np.max(data, axis=0)[0]

    # skewness and kurtosis
    skew_val = skew(data, axis=0)[0]
    kurtosis_val = kurtosis(data, axis=0)[0]

    return mean, std, min_val, max_val, skew_val, kurtosis_val 


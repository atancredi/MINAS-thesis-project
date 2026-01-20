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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from dataloader import load_reflection_spectra_data

    x_train, x_test, y_train, y_test, wavelengths = load_reflection_spectra_data(DATASET_PATH, normalize_spectra=True)

    print(y_train.shape)
    print(y_test.shape)

    # does all the dips starts from 1?
    print("max y train", y_train.flatten().max())
    print("min y train", y_train.flatten().min())
    print("max y test", y_test.flatten().max())
    print("min y test", y_test.flatten().min())

    # show some spectra
    plt.plot(wavelengths, y_train[1].reshape(-1))
    plt.show()
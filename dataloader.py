import pickle
import os

import torch
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from data_analysis import analyze_distribution

# date: 06/2023 @author: P. Wiecha
def load_reflection_spectra_data(path_h5, save_scalers=True, test_fraction=0.05, normalize_spectra=False):

    with h5py.File(path_h5) as f_read:
        R_spec = np.array(f_read['R'], dtype=np.float32)
        geo = np.array(f_read['geo'], dtype=np.float32)
        wavelengths = np.array(f_read['wavelengths'], dtype=np.float32)

    geo_mask = [True, False, True, True, True]

    geo = geo[:,geo_mask]

    # # clip R spectra to 64 points
    # if R_spec.shape[1] > 64:
    #     R_spec = R_spec[:, -64:]

    # if wavelengths.shape[0] > 64:
    #     wavelengths = wavelengths[-64:]

    # if necessary, add a channel dimension to the spectra (keras: channels last)
    if R_spec.shape[-1] != 1:
        R_spec = np.expand_dims(R_spec, -1)

    #  separately standardize permittivities and thicknesses
    scaler_geo = StandardScaler().fit(geo)

    # save the scalers using pickle
    if save_scalers:
        pickle.dump(scaler_geo,
                    open('{}_scalers.pkl'.format(os.path.splitext(path_h5)[0]), 'wb'))

    # apply scaler
    geo = scaler_geo.transform(geo)

    if geo.shape[-1] != 1:
        geo = np.expand_dims(geo, -1)

    # split into training and test datasets. Set random state for a reproducible splitting
    x_train, x_test, y_train, y_test = train_test_split(
        geo, R_spec, test_size=test_fraction, random_state=42)
    

    if normalize_spectra:
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

        train_min_max_scaler = MinMaxScaler()
        y_train = train_min_max_scaler.fit_transform(y_train)
        test_min_max_scaler = MinMaxScaler()
        y_test = test_min_max_scaler.fit_transform(y_test)

        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    return x_train, x_test, y_train, y_test, wavelengths


def print_data_stats(data):
    x_mean, x_std, x_min, x_max, x_skew, x_kurtosis = analyze_distribution(data)
    print(f"mean: {x_mean:.3f} | stdev: {x_std:.3f} | min: {x_min:.3f} | max: {x_max:.3f} | skew: {x_skew:.3f} | kurt: {x_kurtosis:.3f}")


class ReflectivityDataset(torch.utils.data.Dataset):

	def __init__(self, X, y, image=False):
		if not torch.is_tensor(X) and not torch.is_tensor(y):
			X = torch.from_numpy(X)
			y = torch.from_numpy(y)

		# # squeeze
		# X = X.squeeze()
		# y = y.squeeze()

		if image:
			y = y.view(-1,9,9)

		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, i):
		return self.X[i], self.y[i]



def load_reflection_spectra_dataloaders(
    dataset_path: str,
    test_fraction: float = 0.05,
    batch_size = 32,
    normalize_spectra=False
):
    
    x_full_train, x_test, y_full_train, y_test, wavelength = load_reflection_spectra_data(dataset_path, test_fraction=test_fraction, normalize_spectra=normalize_spectra)

    x_train, x_val, y_train, y_val = train_test_split(
        x_full_train, y_full_train, test_size=0.1, random_state=42
    )

    print(f"train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")
    
    train_dataset = ReflectivityDataset(x_train, y_train)
    val_dataset   = ReflectivityDataset(x_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, x_test, y_test, wavelength


import pickle
import os

import torch
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_prominences


# from .data_analysis import analyze_distribution

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

	return x_train, x_test, y_train, y_test, wavelengths, scaler_geo



import numpy as np
import torch
from scipy.signal import find_peaks, peak_prominences

def compute_peak_bounds(signal, k_peaks=3, n_exclude=10, peak_buffer=5):
	inverted_signal = 1 - signal
	all_peaks, _ = find_peaks(inverted_signal)
	
	# Filter window
	window_peaks = all_peaks[(all_peaks >= n_exclude) & (all_peaks < len(signal) - n_exclude)]
	
	if len(window_peaks) == 0:
		return None
		
	# Get top k
	proms = peak_prominences(inverted_signal, window_peaks)[0]
	top_k_indices = np.argsort(proms)[-k_peaks:]
	final_peaks = window_peaks[top_k_indices]
	
	return np.min(final_peaks) - peak_buffer, np.max(final_peaks) + peak_buffer

def apply_smooth_flattening(signal, bounds):
	if bounds is None:
		return signal
	
	left_bound, right_bound = bounds
	x = np.arange(len(signal))
	
	# Sigmoid logic
	k = 0.8
	mask_left = 1 / (1 + np.exp(-k * (x - left_bound)))
	mask_right = 1 / (1 + np.exp(k * (x - right_bound)))
	
	envelope = mask_left * mask_right
	return 1 + (signal - 1) * envelope


class ReflectivityDataset(torch.utils.data.Dataset):
	def __init__(self, X, y, image=False, augment_spectra=False, p_aug=0.3):
		X_np = X.numpy() if torch.is_tensor(X) else X
		y_np = y.numpy() if torch.is_tensor(y) else y

		self.X = torch.from_numpy(X_np).float()
		self.y = torch.from_numpy(y_np).float()
		self.augment_spectra = augment_spectra
		self.p_aug = p_aug

		if image:
			self.y = self.y.view(-1, 9, 9)

		self.peak_bounds = []
		if augment_spectra:
			print("Pre-calculating peaks")
			for i in range(len(self.y)):
				bounds = compute_peak_bounds(y_np[i].squeeze())
				self.peak_bounds.append(bounds)
		else:
			self.peak_bounds = [None] * len(self.y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, i):
		geo = self.X[i]
		signal = self.y[i]

		if self.augment_spectra and np.random.rand() < self.p_aug:
			# Apply flattening using the pre-calculated bounds for this index
			bounds = self.peak_bounds[i]
			aug_signal = apply_smooth_flattening(signal.numpy().squeeze(), bounds)
			signal = torch.from_numpy(aug_signal).float().unsqueeze(-1)

		return geo, signal


def load_reflection_spectra_dataloaders(
	dataset_path: str,
	test_fraction: float = 0.05,
	batch_size = 32,
	augment_spectra=False,
	p_aug=0.3
):
	
	x_full_train, x_test, y_full_train, y_test, wavelength, scaler_geo = load_reflection_spectra_data(dataset_path, test_fraction=test_fraction)

	x_train, x_val, y_train, y_val = train_test_split(
		x_full_train, y_full_train, test_size=0.1, random_state=42
	)

	print(f"train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")
	
	train_dataset = ReflectivityDataset(x_train, y_train, augment_spectra=augment_spectra, p_aug=p_aug)
	val_dataset   = ReflectivityDataset(x_val, y_val, augment_spectra=False, p_aug=0)
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	return train_loader, val_loader, x_test, y_test, wavelength, scaler_geo



if __name__ == "__main__":
	  
	from dotenv import load_dotenv
	from os import getenv
	load_dotenv("params.env")
	dataset_path = getenv("DATASET_PATH")
	train_loader, val_loader, x_test, y_test, wavelength, _ = load_reflection_spectra_dataloaders(dataset_path)

	x,y = next(iter(train_loader))
	x = x[0].squeeze()
	y = y[0].squeeze()
	# print(x,y)

import torch
from torch import nn
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataloader import load_reflection_spectra_data

DATASET_PATH = "__data/pisa_data_2025_12_02.h5"


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
	

class ForwardMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden_activation = nn.LeakyReLU()

		self.layers = nn.Sequential(
			nn.Linear(4, 512),
			self.hidden_activation,
			nn.Linear(512, 256),
			self.hidden_activation,
			nn.Linear(256, 128),
			self.hidden_activation,
			nn.Linear(128, 81)
		)


	def forward(self, x):
		return self.layers(x)
	

if __name__ == '__main__':

	lr = 0.001
	
	# fixed random number seed
	torch.manual_seed(42)

	x_train, x_test, y_train, y_test, wavelength = load_reflection_spectra_data(DATASET_PATH, test_fraction=0.05)
	print(x_train.shape, x_test.shape)
	print(y_train.shape, y_test.shape)
	
	# Load dataset
	dataset = ReflectivityDataset(x_train, y_train)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)


	mlp = ForwardMLP()
	loss_function = nn.MSELoss() # mse loss (regression)
	optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)


	epochs = 5
	arch_version = "01"
	model_name = f"__models/mlp_model_{arch_version}_{epochs}.pth"
	os.makedirs("__models", exist_ok=True)
	if not os.path.exists(model_name):
		for epoch in range(0, epochs):

			print(f'Starting epoch {epoch+1}')
			current_loss = 0.0

			for i, data in enumerate(trainloader):
				
				inputs, targets = data
				inputs = inputs.view(inputs.size(0), -1)
				targets = targets.view(targets.size(0), -1)
				inputs, targets = inputs.float(), targets.float()
				# targets = targets.reshape((targets.shape[0], 1))
				
				# forward pass
				optimizer.zero_grad()
				outputs = mlp(inputs)
				
				# loss
				loss = loss_function(outputs, targets)
				
				# backward pass
				loss.backward()
				
				# optimiz step
				optimizer.step()
				
				# update loss and print
				current_loss += loss.item()
				if i % 100 == 0:
					print('Loss after mini-batch %5d: %.5f' %
						(i + 1, current_loss / 500))
					current_loss = 0.0
		
		print('training finished')
		torch.save(mlp.state_dict(), model_name)
		print('model saved')

	else:
		mlp.load_state_dict(torch.load(model_name))



	# TEST MODEL
	mlp.eval()
	
	x_test_tensor = torch.from_numpy(x_test).float()
	x_test_tensor = x_test_tensor.view(x_test_tensor.size(0), -1)


	# inference
	with torch.no_grad():
		y_pred = mlp(x_test_tensor)

	y_pred = y_pred.numpy()

	
	# y_test = y_test.reshape(-1, 1) 
	y_test = y_test.squeeze()
	# y_pred = y_pred.reshape(-1, 1)
	y_pred = y_pred.squeeze()

	# evaluate performance
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)


	print(f"MSE: {mse} | MAE: {mae} | R2: {r2}")

	
	# y_test_flat = y_test.flatten()  # Flatten to 1D
	# y_pred_flat = y_pred.flatten()  # Flatten to 1D

	# # Plot the ideal line (y = x)

	# import matplotlib.pyplot as plt
	# import seaborn as sns

	# # Scatter plot of actual vs predicted values
	# plt.figure(figsize=(8, 6))
	# sns.scatterplot(x=y_test_flat, y=y_pred_flat)
	# plt.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], color='red', linestyle='--')  # Ideal line
	# plt.xlabel("Actual Values")
	# plt.ylabel("Predicted Values")
	# plt.title("Actual vs Predicted Values")
	# plt.show()


	# # Calculate residuals
	# residuals = y_test_flat - y_pred_flat

	# # Plot residuals
	# plt.figure(figsize=(8, 6))
	# sns.histplot(residuals, kde=True)
	# plt.title("Residuals Distribution")
	# plt.xlabel("Residuals")
	# plt.ylabel("Frequency")
	# plt.show()




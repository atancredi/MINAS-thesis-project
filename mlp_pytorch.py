import torch
from torch import nn

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
	def __init__(self, input_dim=4, output_dim=81, hidden_layers=[1024, 512, 256, 128], activation_name: str = "LeakyReLU"):
		super(ForwardMLP, self).__init__()
		
		activ = getattr(nn, activation_name)
		layers = []
		
		layers.append(nn.Linear(input_dim, hidden_layers[0]))
		layers.append(activ())
		
		for i in range(len(hidden_layers) - 1):
			layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
			layers.append(activ())
			
		layers.append(nn.Linear(hidden_layers[-1], output_dim))
		
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)

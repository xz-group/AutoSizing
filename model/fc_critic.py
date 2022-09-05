"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_critic(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim, n_latent_var):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		super(FeedForwardNN_critic, self).__init__()

		self.layer1 = nn.Linear(in_dim, n_latent_var)
		self.layer2 = nn.Linear(n_latent_var, n_latent_var)
		self.layer3 = nn.Linear(n_latent_var, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		#activation1 = F.tanh(self.layer1(obs))
		activation1 = torch.tanh(self.layer1(obs))
		#activation2 = F.tanh(self.layer2(activation1))
		activation2 = torch.tanh(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
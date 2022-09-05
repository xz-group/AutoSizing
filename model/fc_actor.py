"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_actor(nn.Module):
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
		super(FeedForwardNN_actor, self).__init__()

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
		dim_zero = self.layer3(activation2).shape[0]
		#dim_one = self.layer3(activation2).shape[1]
		if dim_zero == 21:
			output = F.softmax(torch.reshape(self.layer3(activation2), (7, 3)), dim=1)
		else:
			output = F.softmax(torch.reshape(self.layer3(activation2), (dim_zero, 7, 3)), dim=2)

		# 	self.layer3(activation2)
		#output = F.softmax(torch.reshape(self.layer3(activation2), (7, 3)), dim=1)

		# output_0 = F.softmax(self.layer3(activation2)[0:3], dim=-1)
		# output_1 = F.softmax(self.layer3(activation2)[3:6], dim=-1)
		# output_2 = F.softmax(self.layer3(activation2)[6:9], dim=-1)
		# output_3 = F.softmax(self.layer3(activation2)[9:12], dim=-1)
		# output_4 = F.softmax(self.layer3(activation2)[12:15], dim=-1)
		# output_5 = F.softmax(self.layer3(activation2)[15:18], dim=-1)
		# output_6 = F.softmax(self.layer3(activation2)[18:21], dim=-1)
		# output = torch.cat((output_0, output_1,output_2,output_3,output_4,output_5,output_6), dim=0)
		return output
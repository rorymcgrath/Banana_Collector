import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
	'''The Q network used by the agent.'''

	def __init__(self,state_size,action_size,seed):
		'''Initlise and defined the model.

		Parameters
		----------
		state_size : int
			The Dimension of each state.

		action_size : int
			The Dimension of each action.

		seed : int
			The random seed used.
		'''
		super(QNetwork,self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size,148)
		self.fc2 = nn.Linear(148,148)
		self.fc3 = nn.Linear(148,action_size)

	def forward(self,state):
		'''Build the network that estimates Q values for each state.

		Parameters
		----------
		state : array_like
			The current state.


		Returns
		-------
		Q_values : array_like
			The Q_values for each action given the current state.
		'''
		return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))

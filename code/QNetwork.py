import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

	def __init__(self,state_size,action_size,seed):
		super(QNetwork,self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size,state_size*2)
		self.fc2 = nn.Linear(state_size*2,state_size)
		self.fc3 = nn.Linear(state_size,action_size)

	def forward(self,state):
		return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))

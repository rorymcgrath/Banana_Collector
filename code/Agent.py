import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork

LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
GAMMA = 0.99
TAU = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

	def __init__(self, state_size, action_size, seed):
		self.state_size = state_size
		self.action_size = action_size
		random.seed(seed)
		
		#Q-Network
		self.local_qnetwork = QNetwork(state_size, action_size, seed).to(device)
		self.target_qnetwork = QNetwork(state_size, action_size, seed).to(device)
		self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=LEARNING_RATE)

		#Replay Memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)
		
		self.t_step = (self.t_step+1)%UPDATE_EVERY
		if self.t_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.train_model_parameters(experiences)

	
	def get_action(self, state, epsilon=0):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.local_qnetwork.eval()
		with torch.no_grad():
			action_values = self.local_qnetwork(state)
		self.local_qnetwork.train()
		if random.random() > epsilon:
			return np.argmax(action_values.to(device).data.numpy())
		else:
			return random.randint(0, self.action_size-1)

	def train_model_parameters(self, experiences):
		states, actions, rewards, next_states, dones = experiences
		Q_next_states = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
		Q_states = rewards + GAMMA*Q_next_states*(1-dones)
	
		Q_states_estimated = self.local_qnetwork(states).gather(1,actions)
		
		loss = F.mse_loss(Q_states_estimated,Q_states)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self._copy_model_parameters()     

	def _copy_model_parameters(self):
		for target_param, local_param in zip(self.target_qnetwork.parameters(), self.local_qnetwork.parameters()):
			target_param.data.copy_(TAU*local_param.data + (1-TAU) * target_param.data)

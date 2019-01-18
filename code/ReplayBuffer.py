import numpy as np
import random
import torch
from collections import namedtuple, deque

class ReplayBuffer:

	def __init__(self, action_size, buffer_size, batch_size, seed):
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		random.seed(seed)
		self.experience = namedtuple("Experience", field_names=['state','action','reward','next_state','done'])

	def add(self, state, action, reward, next_state, done):
		e = self.experience(state,action,reward,next_state,done)
		self.memory.append(e)

	def sample(self):
		sampled_experiences = random.sample(self.memory,k=self.batch_size)
		states,actions,rewards,next_states,dones = [],[],[],[],[]
		for e in experiences:
			if e is not None:
				states.append(e.state)
				actions.append(e.action)
				rewards.append(e.reward)
				next_states.append(e.next_state)
				dones.append(e.done)
		return (torch.from_numpy(np.vstack(e)).float() for e in [states,actions,rewards,next_states,dones])

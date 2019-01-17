class Agent():

	def __init__(self):
		pass

	def step(self, state, action, reward, next_state, done):
		pass

	def get_action(self, state, epsilon=0):
		pass

	def train_model_parameters(self, experiences, gamma):
		pass

	def _copy_model_parameters(self, local_model, target_model, tau):
		pass

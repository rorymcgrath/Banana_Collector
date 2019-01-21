from unityagents import UnityEnvironment
from Agent import Agent
import numpy as np
from collections import deque
import pickle

MAX_T = 277 
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

env = UnityEnvironment(file_name="../Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

print('Number of agents: {}'.format(len(env_info.agents)))
print('Number of actions: {}'.format(action_size))
print('State Size: {}'.format(state_size))

agent = Agent(state_size,action_size,seed=0)

scores_window = deque(maxlen=100)
scores = []

eps = EPS_START
for i_episode in range(1,5000):
	env_info = env.reset(train_mode=True)[brain_name]
	state = env_info.vector_observations[0]            
	score = 0                                          
	for t in range(MAX_T):
		action = agent.get_action(state,eps)
		env_info = env.step(action)[brain_name]       

		next_state = env_info.vector_observations[0]   
		reward = env_info.rewards[0]                   
		done = env_info.local_done[0]                  
		agent.step(state,action,reward,next_state,done)
		score += reward                                
		state = next_state                             
		if done:                                       
			break
	eps = max(EPS_END,EPS_DECAY*eps)
	scores_window.append(score)
	scores.append(score)

	if i_episode % 100 == 0:
		print('\rEpisode {} \tAverage Score: {:.2f} \tEpsilon: {:.5f}'.format(i_episode, np.mean(scores_window),eps))
	if np.mean(scores_window) >= 10:
		print('Environment soled in {:d} episodes. Average Score: {:.2f} Saving model parameters.'.format(i_episode-100,np.mean(scores_window)))
		torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
		break	
with open('results.pkl','wb') as f:
	pickle.dump(results,f)



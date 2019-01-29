from unityagents import UnityEnvironment
from Agent import Agent
import numpy as np
from collections import deque
import pickle
import torch


#Define the maximum number of time steps in each episode.
MAX_T = 277 
#Define how epsilong will decay during training
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

#Initlise the environment, np_graphics are set to true to disable UI window.
env = UnityEnvironment(file_name="../Banana.app",no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#Initlise the environment for training
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
#Define the minimum aver score for success.
success_score = 13
for i_episode in range(1,1500):
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
	
	if np.mean(scores_window) >= success_score:
		#If the average score is greater then the current success score save the model paremeters and increase the success score by 1
		#Given that the average score is being reposed over 100 episodes the environment was solved 100 episodes previously at the start of this rolling window.
		print('Environment solved in {:d} episodes. Average Score: {:.2f} Saving model parameters.'.format(i_episode-100,np.mean(scores_window)))
		torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
		success_score+=1

with open('scores.pkl','wb') as f:
	#Save the scores so that they can be visualised later.
	pickle.dump(scores,f)


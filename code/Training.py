from unityagents import UnityEnvironment
from Agent import Agent
import numpy as np

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
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]            
score = 0                                          

for i_episode in range(1,10):
	action = agent.get_action(state)
	print(action)
	env_info = env.step(action)[brain_name]       

	next_state = env_info.vector_observations[0]   
	reward = env_info.rewards[0]                   
	done = env_info.local_done[0]                  
	agent.step(state,action,reward,next_state,done)

	score += reward                                
	state = next_state                             
	if done:                                       
		break
    
print("Score: {}".format(score))

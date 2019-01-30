import pickle
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

with open('scores.pkl','rb') as s:
	scores = pickle.load(s)

rewards, = plt.plot([i for i in range(len(scores))],scores, 'b',label='Reward')
average_rewards, = plt.plot([i for i in range(100,len(scores))],[np.mean(scores[i-100:i]) for i in range(100,len(scores))], '--r', label='Averge Reward')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.legend(handles=[rewards,average_rewards])
plt.grid(True)
plt.title('Results')
plt.show()

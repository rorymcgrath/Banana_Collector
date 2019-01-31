[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Banana Collector

### Introduction

This project uses AI gym and a unity environment to train an agent to navigate around a large square environment and collect bananas.
All code relating to the project can be found in the `code` directory.
Trained weights for the ANN's have been saved to the `checkpoint.pth` file.
A report has been created and placed in the `docs` folder. 
This folder contains the .tex and .bib files used to generate the report along the generated .pdf report.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. 
Given this information, the agent learns how to best select actions.  
Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

This task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

In order to get up and running with this codebase the following instructions have been provided.

1. Clone this repository.

2. Download the unity environment from one of the links below.  
You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the unzipped file in the root directory of this cloned repo.

3. It is recommended that you setup anaconda or conda so that you can run the Jupyter notebook and resolve any dependencies that gym or pytorch may have.
	1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
	2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
		- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
		- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

	3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
	```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
	```

	4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

	![Kernel][image2]


	5. If you encounter any additional dependency issues, clone and run the following [python code](https://github.com/udacity/deep-reinforcement-learning/tree/master/python)

### Instructions

Follow the instructions in `Navigation.ipynb` for an explanation of how the code works. Navigate to the `code` directory and execute the following.

```bash
conda activate drlnd
jupyter notebook Navigation.ipynb
```

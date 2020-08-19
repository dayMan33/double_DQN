Double DQN
=====

Contents
--------

* [Description](#description)
* [Instalation](#installation)
* [Usage](#usage)
* [Support](#support)

description
-----------
A double deep Q-learning library implemented in python3 using [tensorflow](https://www.tensorflow.org/)

Q-learning is is a model-free [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) algorithm 
to learn a policy telling an agent what action to take under what circumstances. 
This library provides the basic structure to learn a policy in an environment using the 
[Double-deep Q-learning](https://arxiv.org/pdf/1509.06461.pdf) algorithm.
 
The code files in this project include: 
- **dqn_agent.py:**
- **double_dqn.py:** 
- **experience_replay.py:** 
- **dqn_env.py:** 
 
installation
--------

### clone
Clone this repository to your local machine using 'repository address goes here'
            
    git clone https://github.com/dayMan33/double_DQN.git

### setup 
while in the project directory, run setup.sh to install all requirements.

    double_dqn> setup.sh

usage
-----
To start training an agent, you must implement a class of dqn_env with the required methods. Only then can you 
initialize a dqn_agent with an instance of the environment as its only argument. Once you have done that, you will need
to set the model of the agent to be a compiled tf.keras Model tailored specifically to your environment's needs. 
After setting the agent's model, you can train it by calling dqn_agent.train with the necessary arguments
  
    ```python
    from double_dqn.dqn_env import DQNenv
    from double_dqn.dqn_agent import DQNagent
    
    path = 'model_path'
    num_episodes = N
    env = MyEnv() # Inherits from DQNenv
    agent = DQNagent(env)
    model = build_model(env.get_state_shape(), env.get_action_shape())
    agent.set_model(model) # A compiled tf.keral Model to use as the agent's NN.
    agent.train(num_episodes, path)
     ```
    
The train method saves the weights and the model architecture to the specified path

For a more detailed example, check out this [repository](https://github.com/dayMan33/double_dqn_usage.git)

support
-------
For any questions or comments, feel free to email me at danielrotem33@gmail.com



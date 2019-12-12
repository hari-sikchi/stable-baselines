
from stable_baselines import DQN
# from stable_baselines.common.evaluation import evaluate_policy
import gym
from gym import spaces
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import PPO2, SAC
import hopper_rep
import os
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import json

class RandomWalkEnv(gym.Env):
    def __init__(self,total_states = '100'):
       self.past_actions = []
       print("Total states in this environment is: {}".format(total_states))
       self.metadata={'render.modes': ['human']}
       self.states = np.arange(total_states)
       self.total_states = total_states
       self.reset()
       self.action_space = spaces.Discrete(2)
       self.observation_space = spaces.Discrete(1)
       self.cum_rewards = 0
       self.observation_dim=1
       self.action_dim = 2



    def reset(self):
        # self.task = self.tasks[np.random.choice(len(self.tasks))]
        # print("New task is {}".format(self.task))
        # print("reset called")
        self.state = 0
        self.cum_rewards = 0
        return self.state

    def step(self,action):
        # print("Env state:{}",self.state)
        reward = 0
        done=False
        
        if(self.state == 0 and action==0):
            self.state = self.state
            reward = 0
        elif (self.state==self.total_states-1 and action==1):
            self.state = self.total_states+1
            done=True
            reward=1000
        elif action==0:
            self.state=self.state-1
        elif action==1:
            self.state=self.state+1
            
        self.cum_rewards +=reward
        if(done):
            return self.state,reward,done,{'episode':{'r':self.cum_rewards}}
        return self.state,reward,done,{}
        
 

best_mean_reward, n_steps = -np.inf, 0
max_timesteps=10

test_env = RandomWalkEnv(max_timesteps)
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """


    global n_steps, best_mean_reward
    # Print stats every 1000 calls

    total_reward=0
    mean_reward=0
    if (n_steps + 1) % 1== 0:
        for i in range(1):
            dones=False
            timesteps = 0
            obs = test_env.reset()
            while not dones:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = test_env.step(action)
                total_reward+=rewards
                timesteps+=1
                if(timesteps==max_timesteps*3):
                    dones=True
                if(dones):
                    break
        mean_reward=total_reward
        print("Timesteps required to solve the task:{}".format(timesteps))
        if(timesteps==max_timesteps):
            return False
        n_steps += 1
    # Returning False will stop training early
    return True








# Create environment
env = RandomWalkEnv(max_timesteps)

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True,verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5),log_interval=100, callback=callback)
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_lunar")

# Evaluate the agent
# mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

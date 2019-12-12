
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
seed = 500
set_global_seeds(seed)



class RandomWalkEnv(gym.Env):
    def __init__(self,total_states = '100'):
       self.past_actions = []
    #    print("Total states in this environment is: {}".format(total_states))
       self.metadata={'render.modes': ['human']}
       self.states = np.arange(total_states)
       self.total_states = total_states
       self.reset()
       self.action_space = spaces.Discrete(2)
       self.observation_space =  spaces.Box(low=np.array([-1.0]),high=np.array([1.0]))
       self.cum_rewards = 0
       self.observation_dim=1
       self.action_dim = 2

    def get_int_to_state(self,state):
        normalized_state = state/float(self.total_states)-1
        return normalized_state

    # def get_q(self):
    #     for state in range(self.total_states):
    #         normalized_state = state/float(self.total_states)-1
    #         normalized_state=np.array(normalized_state).reshape(1,1)
    #         print("State: {}, Action prob:{}",state,)

    def reset(self):
        # self.task = self.tasks[np.random.choice(len(self.tasks))]
        # print("New task is {}".format(self.task))
        # print("reset called")
        self.state = 0
        self.cum_rewards = 0
        normalized_state = self.state/float(self.total_states)-1
        return np.array([normalized_state]).reshape(1,)

    def step(self,action):
        # print("Env state:{}",self.state)
        reward = 0
        done=False
        
        # if(self.state == 0 and action==0):
        #     self.state = self.state
        #     reward = 0
        # elif (self.state==self.total_states-1 and action==1):
        #     self.state = self.total_states+1
        #     done=True
        #     reward=1000
        if action==0 and self.state%2==0:
            self.state=self.state-1
        elif action==1 and self.state%2==0:
            self.state=self.state+1

        elif action==0 and self.state%2==1:
            self.state=self.state+1      
        elif action==1 and self.state%2==1:
            self.state=self.state-1
        
        if(self.state<0):
            self.state = 0
        if(self.state>=self.total_states):
            self.state = self.total_states
            reward=1000
            done=True
            
        self.cum_rewards +=reward
        
        normalized_state = self.state/float(self.total_states)-1
        if(done):
            return np.array(normalized_state).reshape(1,),reward,done,{'episode':{'r':self.cum_rewards}}
        return np.array([normalized_state]).reshape(1,),reward,done,{}
        
results = []
 
for horizon in range(1):
    horizon = 2
    best_mean_reward, n_steps = -np.inf, 0
    max_timesteps=horizon+1
    steps_to_solve = 0

    test_env = RandomWalkEnv(max_timesteps)
    print("Starting to learn task with horizon: {}".format(horizon+1))
    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """


        global n_steps, best_mean_reward,max_timesteps,steps_to_solve
        # Print stats every 1000 calls

        total_reward=0
        mean_reward=0
        steps_to_solve+=1
        if (n_steps + 1) % 1== 0:
            for i in range(1):
                dones=False
                timesteps = 0
                obs = test_env.reset()
                while not dones:
                    action, _states = model.predict(np.array(obs).reshape(1,1))
                    print("state: {}, action: {}".format(test_env.state,action))
                    obs, rewards, dones, info = test_env.step(action)
                    total_reward+=rewards
                    timesteps+=1
                    if(timesteps==max_timesteps*3):
                        dones=True
                    if(dones):
                        break
            mean_reward=total_reward
            print(max_timesteps)
            print("****************************")
            for state in range(test_env.total_states):
                normalized_state = state/float(test_env.total_states)-1
                normalized_state=np.array(normalized_state).reshape(1,1)
                print("State: {}, Action prob:{}".format(state,model.action_probability(normalized_state)))
                # print("Q_values:{}".format(model.step_model.q_values(normalized_state)))
            print("****************************")

            if(timesteps==max_timesteps):
                print("Timesteps required to solve the task:{} Horizon: {}".format(steps_to_solve,horizon+1))
                results.append(steps_to_solve)
                return False
        n_steps += 1
        
        # Returning False will stop training early
        return True








    # Create environment
    env = RandomWalkEnv(max_timesteps)

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3,learning_starts=1, prioritized_replay=True,verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(2e5),log_interval=10000, callback=callback)

print(results)
np.save("results_difficult_100_"+str(seed)+".npy",np.array(results))

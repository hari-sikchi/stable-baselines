
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
import matplotlib.pyplot as plt

seed = 500
set_global_seeds(seed)
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_and_save(log_data,log_dir,title="Learning curve"):

    data = log_data
    y1 = np.array(data['eval_loss'])
    y2 = np.array(data['dt'])

    x = np.array(data['iters'])
    # print(x)
    y1 = moving_average(y1, window=50)

    # print(x.shape)
    # print(y1.shape)
    # print(y2.shape)
    
    # Truncate x
    x = x[len(x) - len(y1):]
    # x = x[len(x) - len(y2):]


    fig = plt.figure(title)

    plt.plot(x, y1,label="Eval loss")
    # plt.plot(x, y2,label="Action Repeat")
    
    plt.xlabel('Iters')
    plt.ylabel('Evaluation')
    plt.title(title)
    plt.legend(loc = "upper left")
    plt.savefig(log_dir+"learning_curve.png")
    plt.clf()               
    

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
        ## Simplest task
        
        
        
        ## Harder task
        if(self.state == 0 and action==1):
            self.state = self.state
            reward = 0
        elif(self.state == 0 and action==0):
            if(self.total_states==1):
                done=True
                reward = 1000
            self.state = self.state+1
            

        elif (self.state==self.total_states-1 and action==1):
            self.state = self.total_states+1
            done=True
            reward=1000
        elif action==0:
            self.state=self.state-1
        elif action==1:
            self.state = self.state+1                
        
        
        
        
        ### Most difficult task
        
        

        
        
        # if action==0 and self.state%2==0:
        #     self.state=self.state-1
        # elif action==1 and self.state%2==0:
        #     self.state=self.state+1

        # elif action==0 and self.state%2==1:
        #     self.state=self.state+1      
        # elif action==1 and self.state%2==1:
        #     self.state=self.state-1
        
        # if(self.state<0):
        #     self.state = 0
        # if(self.state>=self.total_states):
        #     self.state = self.total_states
        #     reward=1000
        #     done=True
            
        self.cum_rewards +=reward
        
        normalized_state = self.state/float(self.total_states)-1
        if(done):
            return np.array(normalized_state).reshape(1,),reward,done,{'episode':{'r':self.cum_rewards}}
        return np.array([normalized_state]).reshape(1,),reward,done,{}
        
results = {'horizon':[],'steps':[],'q_values':[]}
train_results = {'iters':[],'eval_loss':[],'dt':[]} 
for horizon in range(1):
    horizon = 10
    best_mean_reward, n_steps = -np.inf, 0
    max_timesteps=horizon+1
    steps_to_solve = 0
    q_horizon= []
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
                    action, q_values = model.predict(np.array(obs).reshape(1,1))
                    # print("state: {}, action: {} | {}".format(test_env.state,action,q_values))
                    obs, rewards, dones, info = test_env.step(action)
                    total_reward+=rewards
                    timesteps+=1
                    if(timesteps==max_timesteps*3):
                        dones=True
                    if(dones):
                        break
            mean_reward=total_reward
            print(timesteps)
            print("****************************")
            state_q = [] 
            
            train_results['iters'].append(n_steps)
            train_results['eval_loss'].append(pow(0.99,timesteps)*1000)
            train_results['dt'].append(model.action_repetition)
            for state in range(test_env.total_states):
                normalized_state = state/float(test_env.total_states)-1
                normalized_state=np.array(normalized_state).reshape(1,1)
                state_q.append(model.predict(np.array(normalized_state).reshape(1,1))[1])
                # q_horizon.append(model.predict(np.array(normalized_state).reshape(1,1))[1])
                # print("State: {}, Action prob:{} | {}".format(state,model.action_probability(normalized_state),model.predict(np.array(normalized_state).reshape(1,1))[1]))
                # print("Q_values:{}".format(model.step_model.q_values(normalized_state)))
            q_horizon.append(state_q)
            print("****************************")

            if(timesteps==max_timesteps):
                print("Timesteps required to solve the task:{} Horizon: {}".format(steps_to_solve,horizon+1))
                # results.append(steps_to_solve)
                results['steps'].append(steps_to_solve)
                #return False
        n_steps += 1
        
        # Returning False will stop training early
        return True



    # Create environment
    env = RandomWalkEnv(max_timesteps)

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3,learning_starts=1, prioritized_replay=True,verbose=1)
    # Train the agent
    model.learn(total_timesteps=100000,log_interval=10000, callback=callback,use_action_repeat = True,action_repeat = 4)
    results['q_values'].append(q_horizon)
    results['horizon'].append(horizon)
    
    
    
log_dir = "logs/horizon/opt_repeat_"+str(seed)+ "/"
os.makedirs(log_dir, exist_ok=True)
print(train_results)
np.save(log_dir+"train_results_"+str(seed)+".npy",np.array(train_results))
plot_and_save(train_results,log_dir)
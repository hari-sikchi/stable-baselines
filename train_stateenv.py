import gym
from gym import spaces
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
class StateEnv(gym.Env):
    def __init__(self,task = 'T1'):
    # metadata = {'render.modes': ['human']}
       self.past_actions = []
       self.metadata={'render.modes': ['human']}
       self.reset()
       self.tasks = ['T1']
       self.action_space = spaces.Discrete(2)
       self.observation_space= spaces.Box(low=np.array([0,0]),high=np.array([3,3]))

    #    self.tasks= ['T1','T2','T3','T4','T5']
       self.golden_traj={
            'T1':[[0,0,0],[1,0,1]],
            'T2':[[0,0,0],[1,0,0]],
            'T3':[[0,0,0],[1,1,0]],
            'T4':[[1,1,1],[0,0,1]],
            'T5':[[1,1,1],[0,0,1]]
        }
       self.task=task

       self.observation_dim=2
       self.action_dim = 2

    def get_int_state(self,state):
        if(state=='s1'):
            return 1
        elif(state=='s2'):
            return 2
        elif(state=='s3'):
            return 3
        elif(state=='dead'):
            return 4
        else:
            return -1


    def reset(self):
        self.state = 's1'
        self.past_actions=[]
        return np.array([self.get_int_state(self.state),0])

    def step(self,action):
        self.past_actions.append(action)
        reward = 0
        done=False
        if(len(self.past_actions)==3):
            if(self.state=='s1'):
                if(self.golden_traj[self.task][0]==self.past_actions):
                    self.state='s2'
                    reward = +2
                else:
                    self.state='dead'
                    reward = -1
                    done=True
            elif (self.state=='s2'):

                if(self.golden_traj[self.task][1]==self.past_actions):
                    # print("golden trajectory")
                    self.state='s3'
                    reward = 10
                    done = True
                else:
                    self.state='dead'
                    reward = -1
                    done=True
            self.past_actions=[]


        return  np.array([self.get_int_state(self.state),len(self.past_actions)]),reward,done,{'episodes':[]}

if __name__ == '__main__':
	n_cpu = 1
	# env = SubprocVecEnv([lambda: StateEnv() for i in range(n_cpu)])


	env = DummyVecEnv([lambda: StateEnv()])

	model = TRPO(MlpPolicy, env, verbose=1)
	model.learn(total_timesteps=40000)
	# model = PPO2(MlpPolicy, env, verbose=1)
	# model.learn(total_timesteps=25000)
	# model.save("ppo2_cartpole")

	# del model # remove to demonstrate saving and loading

	# model = PPO2.load("ppo2_cartpole")

	# # Enjoy trained agent
	# obs = env.reset()
	# while True:
	#     action, _states = model.predict(obs)
	#     obs, rewards, dones, info = env.step(action)
	#     env.render()
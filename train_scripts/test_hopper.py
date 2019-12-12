




import gym
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

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

env = DummyVecEnv([lambda: gym.make("Hopper-v2")])

# Automatically normalize the input features
#env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                           clip_obs=10.)


# model = SAC(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=50000, log_interval=10)
# model.save("sac_pendulum")

# del model # remove to demonstrate saving and loading
env = gym.make("Hopper-v2")

model = SAC(MlpPolicy, env, verbose=1)


model = model.load("logs/mujoco/Hopper_anneal_base/best_model_eval.pkl",env)
model.use_action_repeat = True
print("Action repetition of loaded model is : {}".format(model.action_repetition))
#print(env.obs_rms.mean)

#env = env.load("logs/mujoco/Hopper_normal_1/vec_normalize.pkl",env)
#print(env.obs_rms.mean)
log_data = np.load("logs/mujoco/Hopper_anneal_base/log_data.npy",allow_pickle=True)
# print("Log data :{}".format(log_data.item()))

obs = env.reset()
cum_reward=0
step = 0
max_obs = obs.copy()
min_obs = obs.copy()
obs_list= np.array(obs).reshape(-1,1)
while True:
    action, _states = model.predict(obs)
    prev_obs = obs.copy()
    if model.use_action_repeat:
        for _ in range(1):#model.action_repetition):
            obs, rewards, dones, info = env.step(action)
            cum_reward+=rewards
    else:
        obs, rewards, dones, info = env.step(action)
        cum_reward+=rewards
    obs_list = np.concatenate((obs_list,np.array(obs).reshape(-1,1)),axis=1)
    max_obs = np.maximum(obs,prev_obs)
    min_obs = np.minimum(obs,prev_obs)
    prev_obs= obs
    step+=1
    if(dones):
        print(dones,info)
        obs = env.reset()
        break
    #print(rewards)
    #env.render()

# np.set_printoptions(precision=10)
np.set_printoptions(suppress=True)
print(max_obs)
print(min_obs)
print(np.var(obs_list,axis=1))
print("Reward :{}, steps :{}".format(cum_reward,step))

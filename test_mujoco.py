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

model = model.load("logs/mujoco/Hopper_normal_1/best_model_eval.pkl",env)

#print(env.obs_rms.mean)

#env = env.load("logs/mujoco/Hopper_normal_1/vec_normalize.pkl",env)
#print(env.obs_rms.mean)
print(model)

obs = env.reset()
cum_reward=0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cum_reward+=rewards
    if(dones):
        obs = env.reset()
        break
    #print(rewards)
    #env.render()

print("Reward :{}".format(cum_reward))

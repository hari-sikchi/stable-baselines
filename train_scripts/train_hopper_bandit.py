
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import PPO2, SAC
import hopper_rep
import os
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import json
from stable_baselines.common import set_global_seeds


class UCB1Agent:

    def __init__(self, n_arms, dist='bernoulli', **kwargs):

        self.arms = n_arms
        self.num_iters = 0
        self.dist = dist

        if self.dist == 'bernoulli':

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n_arms
                self.means = kwargs['mean']

            else:
                self.means = np.zeros((n_arms, 1))

            self.samples = np.zeros((n_arms, 1))
            self.best_arm = np.argmax(self.means)

    def play(self):

        self.num_iters += 1

        if self.num_iters <= self.arms:
            self.played_arm = self.num_iters - 1

        else:
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)/self.samples)

            self.played_arm = np.argmax(conf_values)

        return self.played_arm

    def update(self, reward):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1















seed = 400
set_global_seeds(seed)




best_mean_reward, n_steps = -np.inf, 0
best_eval_mean_reward = -np.inf
log_dir = "logs/mujoco/Hopper_dump_"+str(seed)+"/"
# log_dir = "logs/mujoco/Hopper_bandit3/"
os.makedirs(log_dir, exist_ok=True)
log_data = {'dt':[],'eval':[],'train':[],'timesteps':[]}

f = open(log_dir+"eval.txt", "w")

test_env = DummyVecEnv([lambda: gym.make("Hopper-v2")])
max_eval_timesteps = 5000
prev_dt_eval = 0
dt_agent = UCB1Agent(4)
# Automatically normalize the input features
# test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False,
#                         clip_obs=10.)









def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """


    global n_steps, best_mean_reward, best_eval_mean_reward,prev_dt_eval
    # Print stats every 1000 calls
    
    total_reward=0
    mean_reward=0
    if (n_steps + 1) % 1000== 0:
        for i in range(100):
            dones=False
            timesteps = 0
            obs = test_env.reset()
            while not dones:
                action, _states = model.predict(obs)
                if model.use_action_repeat:
                    for _ in range(1):
                        obs, rewards, dones, info = test_env.step(action)
                        total_reward+=rewards
                        timesteps+=1
                        if(timesteps==max_eval_timesteps):
                            dones=True
                        if(dones):
                            break
                else:
                    timesteps+=1
                    obs, rewards, dones, info = test_env.step(action)
                    total_reward+=rewards
                    if(timesteps==max_eval_timesteps):
                        dones=True
                 
                if(dones):
                    break
        mean_reward=total_reward/100.0
        dt_reward = mean_reward-prev_dt_eval
        dt_agent.update(dt_reward/100.0)
        print("Mean reward last dt: {}".format(dt_reward))

        prev_dt_eval=mean_reward
        new_dt = int(dt_agent.play())+1
        model.action_repetition=new_dt
        print("Action repetition is :{}".format(model.action_repetition))

        print("Steps: {} 100 Episode eval: {} Best eval {} ".format(n_steps,mean_reward,best_eval_mean_reward))
        f.write("Steps: {} 100 Episode eval: {} Best eval {}\n".format(n_steps,mean_reward,best_eval_mean_reward))
        if mean_reward > best_eval_mean_reward:
            best_eval_mean_reward = mean_reward
            # Example for saving best model
            print("Saving new best model")
            _locals['self'].save(log_dir + 'best_model_eval.pkl')

        log_data['dt'].append(model.action_repetition)
        log_data['eval'].append(mean_reward)
        log_data['timesteps'].append(model.num_timesteps)

        # Evaluate policy training performance

    if (n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
            log_data['train'].append(mean_reward)
                
                
    
    n_steps += 1
    # Returning False will stop training early
    return True

print("Starting Experiment with seed: {}".format(seed))
# env_s= lambda: gym.make("HopperEnvRep-v0")
# env_s = Monitor(env_s, log_dir, allow_early_resets=True)

env = DummyVecEnv([lambda: gym.make("Hopper-v2")])

# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                            clip_obs=10.)

env = Monitor(env.envs[0], log_dir, allow_early_resets=True)

#env.act_rep = 20

model = SAC(MlpPolicy, env, verbose=1)
#model = PPO2(MlpPolicy, env,verbose=True)

dt = int(dt_agent.play())+1
model.action_repetition=dt
model.learn(total_timesteps=1000000,use_action_repeat= True, callback=callback)
f.close()
# json = json.dumps(log_data)
# f = open(log_dir+"log_data.json","w")
# f.write(json)
# f.close()
np.save(log_dir+"log_data.npy",log_data)

# Don't forget to save the VecNormalize statistics when saving the agent
# log_dir = "logs/hopper_aneal/"
# model.save(log_dir + "sac_hopper")
#env.save(os.path.join(log_dir, "vec_normalize.pkl"))

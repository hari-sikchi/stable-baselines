
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

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0
best_eval_mean_reward = -np.inf
log_dir = "logs/mujoco/Hopper_normal_1/"
os.makedirs(log_dir, exist_ok=True)

f = open(log_dir+"eval.txt", "w")

test_env = DummyVecEnv([lambda: gym.make("Hopper-v2")])

# Automatically normalize the input features
# test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False,
#                         clip_obs=10.)


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """


    global n_steps, best_mean_reward, best_eval_mean_reward
    # Print stats every 1000 calls
    
    total_reward=0
    if (n_steps + 1) % 1000== 0:
        for i in range(100):
            done=False
            obs = test_env.reset()
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = test_env.step(action)
                total_reward+=rewards
                if(dones):
                    break
        mean_reward=total_reward/100.0
        print("Steps: {} 100 Episode eval: {} Best eval {}".format(n_steps,mean_reward,best_eval_mean_reward))
        f.write("Steps: {} 100 Episode eval: {} Best eval {}".format(n_steps,mean_reward,best_eval_mean_reward))
        if mean_reward > best_eval_mean_reward:
            best_eval_mean_reward = mean_reward
            # Example for saving best model
            print("Saving new best model")
            _locals['self'].save(log_dir + 'best_model_eval.pkl')



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
    n_steps += 1
    # Returning False will stop training early
    return True


# env_s= lambda: gym.make("HopperEnvRep-v0")
# env_s = Monitor(env_s, log_dir, allow_early_resets=True)

env = DummyVecEnv([lambda: gym.make("Hopper-v2")])

# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                            clip_obs=10.)

#env = Monitor(env.envs[0], log_dir, allow_early_resets=True)

#env.act_rep = 20

model = SAC(MlpPolicy, env, verbose=1)
#model = PPO2(MlpPolicy, env,verbose=True)
model.learn(total_timesteps=1000000, callback=callback)
f.close()
# Don't forget to save the VecNormalize statistics when saving the agent
# log_dir = "logs/hopper_aneal/"
# model.save(log_dir + "sac_hopper")
#env.save(os.path.join(log_dir, "vec_normalize.pkl"))



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
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import json
best_mean_reward, n_steps = -np.inf, 0
best_eval_mean_reward = -np.inf
seed = 500 
log_dir = "logs/mujoco/Swimmer_dump_"+str(seed)+ "/"
os.makedirs(log_dir, exist_ok=True)
log_data = {'dt':[],'eval':[],'train':[],'timesteps':[]}

f = open(log_dir+"eval.txt", "w")
set_global_seeds(seed)
test_env = DummyVecEnv([lambda: gym.make("Swimmer-v2")])
max_eval_timesteps = 5000
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
        print("Steps: {} 100 Episode eval: {} Best eval {} ".format(n_steps,mean_reward,best_eval_mean_reward))
        f.write("Steps: {} 100 Episode eval: {} Best eval {}\n".format(n_steps,mean_reward,best_eval_mean_reward))
        if mean_reward > best_eval_mean_reward:
            best_eval_mean_reward = mean_reward
            # Example for saving best model
            print("Saving new best model")
            _locals['self'].save(log_dir + 'best_model_eval.pkl')
        print("dt: {}".format(model.action_repetition))
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


# env_s= lambda: gym.make("HopperEnvRep-v0")
# env_s = Monitor(env_s, log_dir, allow_early_resets=True)

env = DummyVecEnv([lambda: gym.make("Swimmer-v2")])

# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                            clip_obs=10.)
env.envs[0].seed(seed)
env = Monitor(env.envs[0], log_dir, allow_early_resets=True)

#env.act_rep = 20

model = SAC(MlpPolicy, env, verbose=1)
print("Starting Experiment with seed: {}".format(seed))

#model = PPO2(MlpPolicy, env,verbose=True)
model.learn(total_timesteps=3000,use_action_repeat= True,poisson=False, callback=callback,only_explore_with_act_rep = False)
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








# import gym
# import numpy as np

# from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import PPO2, SAC
# import hopper_rep
# import os
# import gym
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# from stable_baselines.bench import Monitor
# from stable_baselines.results_plotter import load_results, ts2xy

# env = DummyVecEnv([lambda: gym.make("Swimmer-v2")])

# # Automatically normalize the input features
# #env = VecNormalize(env, norm_obs=True, norm_reward=False,
# #                           clip_obs=10.)


# # model = SAC(MlpPolicy, env, verbose=1)
# # model.learn(total_timesteps=50000, log_interval=10)
# # model.save("sac_pendulum")

# # del model # remove to demonstrate saving and loading
# env = gym.make("Swimmer-v2")

# model = SAC(MlpPolicy, env, verbose=1)


# model = model.load("logs/mujoco/Swimmer_normal/best_model_eval.pkl",env)
# model.use_action_repeat = True
# print("Action repetition of loaded model is : {}".format(model.action_repetition))
# #print(env.obs_rms.mean)

# #env = env.load("logs/mujoco/Hopper_normal_1/vec_normalize.pkl",env)
# #print(env.obs_rms.mean)
# log_data = np.load("logs/mujoco/Swimmer_normal/log_data.npy",allow_pickle=True)
# # print("Log data :{}".format(log_data.item()))

# obs = env.reset()
# cum_reward=0
# step = 0
# while True:
#     action, _states = model.predict(obs)

#     if model.use_action_repeat:
#         for _ in range(1):#model.action_repetition):
#             obs, rewards, dones, info = env.step(action)
#             cum_reward+=rewards
#     else:
#         obs, rewards, dones, info = env.step(action)
#         cum_reward+=rewards
#     step+=1
#     if(dones):
#         print(dones,info)
#         obs = env.reset()
#         break
#     #print(rewards)
#     #env.render()

# print("Reward :{}, steps :{}".format(cum_reward,step))

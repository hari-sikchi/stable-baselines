import sys
import time
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger


def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.original_gamma = gamma
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None
        self.action_repetition=4
        self.poisson = False

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None,use_action_repeat = False,poisson=False,skip_q_curriculum = False,
              state_space_discretization = False,discretization_bounds=100000,only_explore_with_act_rep = False):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        self.use_action_repeat=use_action_repeat
        # self.action_repetition = 0.8
        self.skip_q_curriculum=skip_q_curriculum
        self.running_action_repetition = self.action_repetition
        self.state_space_discretization = state_space_discretization
        self.poisson=poisson
        self.poisson_action = 4
        self.poisson_mean = 4
        self.replay_buffer2 = ReplayBuffer(self.buffer_size)
        self.replay_buffer1 = ReplayBuffer(self.buffer_size)
        self.skip_q = 5
        self.discretization_bounds = discretization_bounds
        self.running_skip_q=self.skip_q
        prev_action = None
        # self.prob_past = 0.6
            #self.env.act_rep-=(21-4)/float(total_timesteps)
        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            # if(poisson):
            #     np.concatenate((obs,))
                
            # print(obs)
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []
            self.num_timesteps=0

            for step in range(total_timesteps):
                if poisson:
                    if(self.poisson_mean<1):
                        self.poisson_mean=1

                    self.poisson_action = int(np.random.poisson(self.poisson_mean))

                    self.poisson_mean-=((5)/float(total_timesteps))
                    if(self.poisson_action<1):
                        self.poisson_action=1
                        
                
                if self.skip_q_curriculum:
                    amount = ((4)/float(total_timesteps))
                    self.running_skip_q -=   amount
                    self.skip_q =  int(self.running_skip_q)
                    if(self.skip_q<=1):
                        self.skip_q=1         
            
                if use_action_repeat:
                    # self.action_repetition-=((0.9)/float(total_timesteps))
                    if only_explore_with_act_rep:
                        pass
                    else:
                        amount = ((4)/float(total_timesteps))
                        self.running_action_repetition -= amount
                        # print("Action repetition is :{}".format(self.action_repetition))
                        # if(self.running_action_repetition<=2 and self.running_action_repetition>1):
                        if(self.num_timesteps>=(total_timesteps/3.0) and self.num_timesteps<(2*total_timesteps/3.0)):
                            if(self.action_repetition==4):
                                # with self.sess.as_default():
                                #     init_new_vars_op = tf.initialize_variables([self.log_ent_coef])
                                #     self.sess.run(init_new_vars_op)
                                print("Flushing replay buffer 4, {} prev_size: {} new size: {}".format(self.action_repetition,len(self.replay_buffer),len(self.replay_buffer2)))
                                # self.replay_buffer = self.replay_buffer2
                                # for tup in self.replay_buffer2._storage:
                                #     self.replay_buffer.add(*tup)

                            self.action_repetition=2
                        if(self.num_timesteps>=(2*total_timesteps/3.0)):
                        # if(self.running_action_repetition<=1):
                            if(self.action_repetition==2):
                                # with self.sess.as_default():
                                #     init_new_vars_op = tf.initialize_variables([self.log_ent_coef])
                                #     self.sess.run(init_new_vars_op)
                                print("Flushing replay buffer 2, {} prev_size: {} new size: {}".format(self.action_repetition,len(self.replay_buffer),len(self.replay_buffer1)))
                                # for tup in self.replay_buffer1._storage:
                                #     self.replay_buffer.add(*tup)
                                print(len(self.replay_buffer))
                                # self.replay_buffer = self.replay_buffer1
                            self.action_repetition=1
                        # self.action_repetition = (self.action_repetition*amount +self.action_repetition-amount)/(1-amount+amount*self.action_repetition)
                        # if(self.action_repetition<0):
                        #     self.action_repetition=0
                        # self.env.dec_act_rep((21-4)/float(total_timesteps))
                        # self.running_action_repetition -= ((6-1)/float(total_timesteps))
                    
                        # self.action_repetition = int(self.running_action_repetition)
                        if(self.action_repetition<1):
                            self.action_repetition=1
                        # self.gamma = pow(self.original_gamma,self.action_repetition)    

                    
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # print(self.env.action_space.low,self.env.action_space.high)
                if self.skip_q_curriculum:
                    repeated_reward = 0
                    obs_present= obs.copy()
                    # print("Skipping in Q space for {} values".format(self.skip_q))
                    for act_rep in range(self.skip_q):                        
                        if (self.num_timesteps < self.learning_starts
                            or np.random.rand() < self.random_exploration):
                            # No need to rescale when sampling random action
                            rescaled_action = action = self.env.action_space.sample()
                        else:
                            if poisson:
                                action = self.policy_tf.step(np.concatenate((obs,np.array([self.poisson_action])))[None], deterministic=False).flatten()
                            else:    
                                action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                            # Add noise to the action (improve exploration,
                            # not needed in general)
                            if self.action_noise is not None:
                                action = np.clip(action + self.action_noise(), -1, 1)
                            # Rescale from [-1, 1] to the correct bounds
                            rescaled_action = action * np.abs(self.action_space.low)

                        # if use_action_repeat and prev_action is not None:
                        #     if(np.random.uniform(0,1)<self.action_repetition):
                        #         rescaled_action=prev_action
                        
                        assert action.shape == self.env.action_space.shape                

                        new_obs, reward, done, info = self.env.step(rescaled_action)
                        obs = new_obs
                        repeated_reward+=reward
                        if done:
                            break
                    
                    reward = repeated_reward
                    self.replay_buffer.add(obs_present, action, reward, new_obs, float(done))
                else:                
                
                
                
                
                    # Before training starts, randomly sample actions
                    # from a uniform distribution for better exploration.
                    # Afterwards, use the learned policy
                    # if random_exploration is set to 0 (normal setting)
                    if (self.num_timesteps < self.learning_starts
                        or np.random.rand() < self.random_exploration):
                        # No need to rescale when sampling random action
                        rescaled_action = action = self.env.action_space.sample()
                    else:
                        if poisson:
                            action = self.policy_tf.step(np.concatenate((obs,np.array([self.poisson_action])))[None], deterministic=False).flatten()
                        else:    
                            action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                        # Add noise to the action (improve exploration,
                        # not needed in general)
                        if self.action_noise is not None:
                            action = np.clip(action + self.action_noise(), -1, 1)
                        # Rescale from [-1, 1] to the correct bounds
                        rescaled_action = action * np.abs(self.action_space.low)

                    # if use_action_repeat and prev_action is not None:
                    #     if(np.random.uniform(0,1)<self.action_repetition):
                    #         rescaled_action=prev_action
                    
                    assert action.shape == self.env.action_space.shape
                    
                    # Add action repetition
                    inter_obs = obs.copy()
                    inter_obs2 = obs.copy()
                    inter_reward2 = 0
                    # print("Action repetition is {}".format(self.action_repetition))
                    # print(obs)
                    discretized_obs = ((((np.array(obs)+5))/10.)*self.discretization_bounds).astype(np.int32)
                    # print("Discretized obs: {}".format(discretized_obs))
                    if self.state_space_discretization:
                        
                        repeated_reward = 0
                        max_box_timesteps = 1000
                        box_steps= 0
                        while True:
                            # print("Repeating actions for: {}".format(self.action_repetition))
                            prev_action = rescaled_action
                            new_obs, reward, done, info = self.env.step(rescaled_action)
                            repeated_reward+=reward
                            box_steps+=1
                            new_discretized_obs = (((np.array(new_obs)+5)/10.)*self.discretization_bounds).astype(np.int32)
                            # print("Discretized obs: {}".format(discretized_obs))
                            # print("New Discretized obs: {}".format(new_discretized_obs))
                            array_equal = True
                            for indx, el in enumerate(new_discretized_obs[:-2]):
                                if(el!=discretized_obs[indx]):
                                    # print(indx,discretized_obs[indx])
                                    array_equal=False
                                    break
                                
                            if(not(array_equal) or max_box_timesteps<=box_steps):
                                # print(not(array_equal),max_box_timesteps<=box_steps)
                                # print("break")
                                break
                            if done:
                                break
                        reward = repeated_reward
                                                     
                        
                        
                    
                    elif self.use_action_repeat: 
                        repeated_reward = 0
                        obs_list = [obs]
                        reward_list = []
                        done_list = []
                        
                        
                        if not only_explore_with_act_rep:
                            repeated_reward = 0 
                            for act_rep in range(self.action_repetition):
                                # print("Repeating actions for: {}".format(self.action_repetition))
                                prev_action = rescaled_action
                                new_obs, reward, done, info = self.env.step(rescaled_action)
                                repeated_reward+=reward
                                inter_reward2+=reward#*pow(self.original_gamma,(act_rep)%2)
                                if(act_rep%1==0):
                                    # self.replay_buffer1.add(inter_obs, action, reward, new_obs, float(done))
                                    inter_obs=new_obs
                                if((act_rep+1)%2==0):
                                    # self.replay_buffer2.add(inter_obs2, action, inter_reward2, new_obs, float(done))
                                    inter_obs2=new_obs
                                    inter_reward2=0
                                  
                                if(self.action_repetition==4 and (act_rep+1)%4==0):
                                    self.replay_buffer.add(inter_obs2, action, inter_reward2, new_obs, float(done))
                                    # print("Adding the second half for action rep 4")
                                elif(self.action_repetition==2 and (act_rep+1)%2==0):
                                    self.replay_buffer.add(inter_obs, action, reward, new_obs, float(done))
                                    # print("Adding the second half for action rep 2")
                                if(done):
                                    break
                                                              
                            self.replay_buffer.add(obs, action, repeated_reward, new_obs, float(done))
                            
                                                                           
                            # for act_rep in range(self.action_repetition):
                            #     new_obs, reward, done, info = self.env.step(rescaled_action)
                            #     obs_list.append(new_obs)
                            #     reward_list.append(reward)
                            #     done_list.append(done)
                            #     if done:
                            #         break
                                

                            # for i,obs_el in enumerate(obs_list):
                            #     rep_rew = 0
                            #     gamma_rep = self.original_gamma
                            #     for j in range(i+1,len(obs_list)):
                            #         next_obs_el = obs_list[j]
                            #         rep_rew += pow(gamma_rep,j-i-1)*reward_list[j-1]
                            #         # print(j-i-1)
                            #         self.replay_buffer.add(obs_el, action, rep_rew, next_obs_el, float(done_list[j-1]))
                                    

                        else:
                            # Persistent MDP
                            prev_obs= obs
                            act_reps = np.array([1,2,4])
                            act_rep_sample = int(np.random.choice(act_reps,p=[0.7,0.15,0.15]))
                            # print(act_rep_sample)
                            repeated_reward = 0
                            for act_rep in range(act_rep_sample):
                                new_obs, reward, done, info = self.env.step(rescaled_action)
                                # self.replay_buffer.add(prev_obs, action, reward, new_obs, float(done))
                                repeated_reward+=reward
                                prev_obs = new_obs
                                if done:
                                    break
                            self.replay_buffer.add(obs, action, repeated_reward, new_obs, float(done))
                                        
                            

                        
                        # for act_rep in range(self.action_repetition):
                        #     # print("Repeating actions for: {}".format(self.action_repetition))
                        #     prev_action = rescaled_action
                        #     new_obs, reward, done, info = self.env.step(rescaled_action)
                        #     inter_reward2+=reward#*pow(self.original_gamma,(act_rep)%2)
                        #     if(act_rep%1==0):
                        #         self.replay_buffer1.add(inter_obs, action, reward, new_obs, float(done))
                        #         inter_obs=new_obs
                        #     if((act_rep+1)%2==0):
                        #         self.replay_buffer2.add(inter_obs2, action, inter_reward2, new_obs, float(done))
                        #         inter_obs2=new_obs
                        #         inter_reward2=0
                                
                                
                                
                        #     repeated_reward+=reward#*pow(self.original_gamma,act_rep)
                        #     if done:
                        #         break
                        # reward = repeated_reward
                             
                    elif poisson:
                        repeated_reward = 0
                        # print("Poisson repetition is {}".format(self.poisson_action))
                        for _ in range(self.poisson_action):
                            # print("Repeating actions for: {}".format(self.action_repetition))
                            prev_action = rescaled_action
                            new_obs, reward, done, info = self.env.step(rescaled_action)
                            repeated_reward+=reward
                            if done:
                                break
                        reward = repeated_reward
                        
                    else:
                        new_obs, reward, done, info = self.env.step(rescaled_action)
                    
                    # Store transition in the replay buffer.
                    if poisson:
                        self.replay_buffer.add(np.concatenate((obs,np.array([self.poisson_action]))), action, reward, np.concatenate((new_obs,np.array([self.poisson_action]))), float(done))
                    elif not self.use_action_repeat:   
                        self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                    
                    
                    
                    
                
                
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)

                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
                if done:
                    prev_action=None
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        if self.use_action_repeat:
            data = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
                "target_entropy": self.target_entropy,
                # Should we also store the replay buffer?
                # this may lead to high memory usage
                # with all transition inside
                # "replay_buffer": self.replay_buffer
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "action_noise": self.action_noise,
                "random_exploration": self.random_exploration,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
                "action_repetition": self.action_repetition
            }
        elif self.skip_q_curriculum:
            data = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
                "target_entropy": self.target_entropy,
                # Should we also store the replay buffer?
                # this may lead to high memory usage
                # with all transition inside
                # "replay_buffer": self.replay_buffer
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "action_noise": self.action_noise,
                "random_exploration": self.random_exploration,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
                "skip_q": self.skip_q
            }            
            
            
        elif self.poisson:
            data = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
                "target_entropy": self.target_entropy,
                # Should we also store the replay buffer?
                # this may lead to high memory usage
                # with all transition inside
                # "replay_buffer": self.replay_buffer
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "action_noise": self.action_noise,
                "random_exploration": self.random_exploration,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
                "poisson_mean": self.poisson_mean,
                "poisson_action": self.poisson_action,
            }            
        else:
            data = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
                "target_entropy": self.target_entropy,
                # Should we also store the replay buffer?
                # this may lead to high memory usage
                # with all transition inside
                # "replay_buffer": self.replay_buffer
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "action_noise": self.action_noise,
                "random_exploration": self.random_exploration,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs          
            }            

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

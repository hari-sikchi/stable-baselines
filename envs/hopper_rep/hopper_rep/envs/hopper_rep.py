import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env,hopper


class HopperEnvRep(hopper.HopperEnv):
    def __init__(self,act_rep=20):
        self.act_rep = act_rep
        self.running_act_rep = act_rep
        hopper.HopperEnv.__init__(self)
    def set_act_rep(self, act_rep):
        self.act_rep=act_rep

    def dec_act_rep(self, decrease_rep):
        self.running_act_rep-=decrease_rep
        self.act_rep = int(self.running_act_rep)
        if(self.act_rep<15):
            self.act_rep=15
            
        # if(self.running_act_rep<=self.act_rep/2):
        #     self.act_rep=int(self.act_rep/2.0)
        # # self.act_rep=int(self.running_act_rep)
        # if(self.act_rep<=5):
        #     self.act_rep=5

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        # self.frame_skip=self.act_rep
        # print("Repeating action for {}".format(self.act_rep))

        self.do_simulation(a, self.act_rep)

        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}


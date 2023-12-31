import torch
import gym
from .legged_robot import LeggedRobot

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env:LeggedRobot):
        super().__init__(env)
        self.env = env
        self.obs_history_length = self.env.cfg.env.num_observation_history
        
        self.num_obs_history = self.obs_history_length * self.env.num_obs

        #! 这里选择 stacked 的 history 而不是 concat 的 history
        self.obs_history = torch.zeros(self.env.num_envs, self.obs_history_length,self.env.num_obs, 
                                       dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

    def step(self, action):
        # privileged information and observation history are stored in info
        obs,privileged_obs, rew, done, info = self.env.step(action)
        new_ids = (done > 0).nonzero(as_tuple=False)
        self.obs_history[new_ids, :] = 0
        self.obs_history = torch.cat((self.obs_history[:, 1:], obs.unsqueeze(1)), dim=1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 
                'obs_history': self.obs_history,
                }, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        # self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        self.obs_history = torch.cat((self.obs_history[:, 1:], obs.unsqueeze(1)), dim=1)
        return {'obs': obs, 
                'privileged_obs': privileged_obs, 
                'obs_history': self.obs_history,
                }

    def reset_history(self, env_ids):  # it might be a problem that this isn't getting called!!
        self.obs_history[env_ids, :] = 0

    def reset(self):
        obs, privileged_obs = self.env.reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        self.obs_history = torch.cat((self.obs_history[:, 1:], obs.unsqueeze(1)), dim=1)
        return {"obs": obs, 
                "privileged_obs": privileged_obs, 
                "obs_history": self.obs_history,
                }

    


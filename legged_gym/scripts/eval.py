# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, class_to_dict
from legged_gym.envs.go1.go1_eval_config import Go1Eval, Go1RoughCfgPPOPriLipsNet
from legged_gym.envs.go1.go1_config import Go1RoughCfgPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.runners.on_policy_runner_old import OnPolicyRunner as OnPolicyRunnerOld
import numpy as np
from legged_gym.utils.helpers import update_cfg_from_args
import torch

# noise level
def dump_results(dst:dict, src:dict):
    for k,v in src.items():
        if k in dst.keys():
            dst[k].append(v)
        else:
            dst[k] = [v]
    return dst 



def eval_noise_level(args,
                     use_expert,
                     use_student,
                     model_path,
                    noise_level, 
                    target_vel = [0.5,0.0,0.0],
                    eval_path = "logs/Eval"):
    env_cfg = Go1Eval()
    if use_expert:
        train_cfg = Go1RoughCfgPPO()
    else:
        train_cfg = Go1RoughCfgPPOPriLipsNet()
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1000
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = noise_level
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    env.set_eval(True)
    env.reset()
    env.set_commands(target_vel)
    obs_dict = env.get_observations()
    
    eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + "-noise_level-" + str(noise_level) 
    eval_res = {}
    eval_res['eval_name'] = eval_name
    # load policy
    train_cfg.runner.resume = False
    _,train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg_dict = class_to_dict(train_cfg)
    if use_expert:
        ppo_runner = OnPolicyRunnerOld(env=env, train_cfg=train_cfg_dict,log_dir=None,device = args.rl_device)
    else:
        ppo_runner = OnPolicyRunner(env=env, train_cfg=train_cfg_dict,log_dir=None,device = args.rl_device)
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.load(model_path)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # with torch.inference_mode():
    with torch.no_grad():
        for i in range(int(env.max_episode_length) + 10):
            if use_expert:
                actions = policy(obs_dict)
            else:
                actions = policy(obs_dict,use_student=True)
            obs_dict, rewards, dones, infos = env.step(actions.detach())
            eval_per_step = env.get_eval_result()
            eval_res = dump_results(eval_res, eval_per_step)
    
    for k,v in eval_res.items():
        if type(v) == list:
            eval_res[k] = np.stack(v, axis=1) # (n_env, n_step)
    first_done = np.argmax(eval_res['done'], axis = 1)
    eval_res['first_done'] = first_done
    eval_res['Fall'] = first_done < 1000

    if os.path.exists(eval_path) == False:
        os.makedirs(eval_path)
    eval_file_name = os.path.join(eval_path,eval_name)
    np.save(eval_file_name, eval_res)

    print("Eval Done")

def eval_push_level(args, push_level,push_index=0, push_interval=100, 
                    target_vel = [0.5,0.0,0.0],eval_path = "logs/Eval"):
    env_cfg = Go1Eval()
    train_cfg = Go1RoughCfgPPOPriLipsNet()
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1000 
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_level = push_level
    env_cfg.noise.add_noise = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    env.set_eval(True)
    env.reset()
    env.set_commands(target_vel)
    obs_dict = env.get_observations()
    
    eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + f"-push_level:{push_level}-" + f"-index:{push_index}-" + f"-interval:{push_interval}"
    eval_res = {}
    eval_res['eval_name'] = eval_name
    # load policy
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.load("logs/Multi_K/Nov11_11-19-04_/model_10000.pt")
    policy = ppo_runner.get_inference_policy(device=env.device)
    use_student = False
    # with torch.inference_mode():
    with torch.no_grad():
        for i in range(int(env.max_episode_length)):
            
            if use_student:
                actions = policy(obs_dict,use_student=use_student)
            else:
                actions = policy(obs_dict,use_student=use_student)
            if (i-99) % int(push_interval) == 0:
                obs_dict, rewards, dones, infos = env.step_push(actions.detach(), push_level, push_index)
            else:
                obs_dict, rewards, dones, infos = env.step(actions.detach())
            eval_per_step = env.get_eval_result()
            eval_res = dump_results(eval_res, eval_per_step)
    
    if os.path.exists(eval_path) == False:
        os.makedirs(eval_path)
    eval_file_name = os.path.join(eval_path,eval_name)
    np.savez_compressed

if __name__ == '__main__':
    args = get_args()
    #! Eval Expert 
    # eval_noise_level(args,
    #                  use_expert=True,
    #                  use_student=False,
    #                  model_path="logs/Expert/Nov14_10-10-14_MLP/model_10000.pt",
    #                   noise_level=1, target_vel=[1.0,0.0,0.0],eval_path = "logs/Eval")
    #! Eval MLP Student 
    eval_noise_level(args,
                     use_expert=False,
                     use_student=True,
                     model_path="logs/LipsNet/Nov13_22-04-58_1e-3_L2/model_10000.pt",
                      noise_level=1, target_vel=[1.0,0.0,0.0],eval_path = "logs/Eval")
    # eval_push_level(args, push_level=5, target_vel=[0.5,0.0,0.0],eval_path = "logs/Eval")

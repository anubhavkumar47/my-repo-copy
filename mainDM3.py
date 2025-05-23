import argparse
import math
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from diffusion import Diffusion
from model import TransformerDenoiser, DoubleCritic
from policy import DiffusionOPT
from UAV import Environment


def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-len', type=int, default=10,
                        help="Number of past actions to feed into the Transformer")
    parser.add_argument("--exploration-noise", type=float, default=0.01) # default=0.01
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1e6)#1e6
    parser.add_argument('-e', '--epoch', type=int, default=1e6)# 1000
    parser.add_argument('--step-per-epoch', type=int, default=1)# 100
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
         '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str, default='cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=5)  # for diffusion chain 3 & 8 & 12
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])

    # whether the expert action is availiable
    parser.add_argument('--expert-coef', default=True)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)#0.6
    parser.add_argument('--prior-beta', type=float, default=0.4)#0.4

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args

def main(args=None):
    args = args or get_args()

    # create environment
    env = Environment()
    args.state_shape  = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action   = 1.0

    # Scale exploration noise
    args.exploration_noise *= args.max_action

    # create actor network (history-aware)
    actor_net = TransformerDenoiser(
    state_dim   = args.state_shape,
    action_dim  = args.action_shape,
    hidden_dim  = 128,
    n_heads     = 4,
    n_layers    = 3,
    dropout     = 0.1,
    history_len = args.history_len,
     ).to(args.device)

    # wrap actor in diffusion process
    actor = Diffusion(
        state_dim     = args.state_shape,
        action_dim    = args.action_shape,
        model         = actor_net,
        max_action    = args.max_action,
        beta_schedule = args.beta_schedule,
        n_timesteps   = args.n_timesteps,
        clip_denoised  = True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(),
                                   lr=args.actor_lr,
                                   weight_decay=args.wd)

    # create twin critics
    critic1 = DoubleCritic(
        state_dim  = args.state_shape,
        action_dim = args.action_shape
    ).to(args.device)
    critic_optim1 = torch.optim.Adam(critic1.parameters(),
                                     lr=args.critic_lr,
                                     weight_decay=args.wd)

    critic2 = DoubleCritic(
        state_dim  = args.state_shape,
        action_dim = args.action_shape
    ).to(args.device)
    critic_optim2 = torch.optim.Adam(critic2.parameters(),
                                     lr=args.critic_lr,
                                     weight_decay=args.wd)

    # policy with history support
    policy = DiffusionOPT(
        state_dim      = args.state_shape,
        actor          = actor,
        actor_optim    = actor_optim,
        action_dim     = args.action_shape,
        critic1        = critic1,
        critic_optim1  = critic_optim1,
        critic2        = critic2,
        critic_optim2  = critic_optim2,
        device         = args.device,
        tau            = args.tau,
        gamma          = args.gamma,
        estimation_step= args.n_step,
        lr_decay       = args.lr_decay,
        lr_maxt        = args.epoch,
        expert_coef    = args.expert_coef,
        history_len    = args.history_len,     # <-- pass history_len here too
    )

    examiner = llmExaminer(verbose=True)

    total_steps = 0
    max_episode_steps = 150
    total_episode_csv = []
    total_reward_csv =[]
    total_actor_enegry_csv =[]
    total_critic_csv  =[]
    total_energy_con_csv =[]
    total_aoi_csv =[]
    total_actor_loss=0
    total_critic_loss=0
    



    for i_episode in range(6000):
        state = env.reset()
        state = np.asarray(state, dtype=np.float32)

        # initialize action history deque with zeros
        action_history = deque(
            [np.zeros(args.action_shape, dtype=np.float32)
             for _ in range(args.history_len)],
            maxlen=args.history_len
        )

        epsilon = math.exp(-1. * i_episode / 30)  # annealing

        Reward = 0
        total_energy =0
        total_aoi=0
        t = 0
        done = False

        while not done and t < max_episode_steps:
            # select action (with history)
            if random.random() < epsilon:
                action = np.random.uniform(
                    low=-1, high=1, size=args.action_shape
                ).astype(np.float32)
            else:
                # convert history to array [H, action_dim]
                hist_arr = np.stack(action_history, axis=0)
                action = policy.select_action(state, hist_arr)

            # step environment
            next_state, base_reward, done, dict_element = env.step(action)
            aoi =dict_element["AoI"]
            total_aoi += aoi
            energy =dict_element["Energy Consumption"]
            total_energy +=energy
            #print(dict_element["AoI"])

            # LLM‐shaped reward
            #if t % 10 == 0:
               # llm_score, _ = examiner.evaluate(state, action, base_reward)
            #else:
                #llm_score = 0
            #reward = 0.9 * base_reward + 0.1 * llm_score

            next_state = np.asarray(next_state, dtype=np.float32)
            Reward += base_reward
            #print("Base Reward {}".format(base_reward))
            #print("Next State {}".format(next_state))
            t += 1

            # add to replay buffer
            policy.add_samples(state, action, base_reward, next_state, terminal=done)

            # update history
            action_history.append(action)

            state = next_state
            total_steps += 1

            if total_steps > policy.batch_size*5:
                # print("Update")
                # print(actor_net.get_params())
                loss = policy.learn(t)
                # print(actor_net.get_params())
                critic_loss = loss.get("loss/critic")
                actor_loss = loss.get("overall_loss")
                if critic_loss is None :
                  critic_loss=0
                if  actor_loss is None:
                  actor_loss =0

                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                # print(loss.get("loss/critic"),loss.get("overall_loss"))

        print(f"Episode {i_episode}: Avg Reward per step = {Reward / t:.3f}")
        print(f"Episode {i_episode}: Critic Loss = {total_critic_loss/t:.3f}")
        print(f"Episode {i_episode}: Actor Loss = {total_actor_loss/t:.3f}")
        print(f"Episode {i_episode}: AoI= {total_aoi / t:.3f}")
        print(f"Episode {i_episode}:Energy = {total_energy[0]/ t}")
        #print(f"Episode {i_episode}: Avg Reward per step = {Reward / t:.3f}")
        total_episode_csv.append(i_episode)
        total_reward_csv.append(Reward/t)
        total_critic_csv.append(total_critic_loss/t)
        total_actor_enegry_csv.append(total_actor_loss/t)
        total_aoi_csv.append(total_aoi/t)
        total_energy_con_csv.append(total_energy[0]/ t)
       
        print("-" * 80)
    data_frame = pd.DataFrame({"Episode":total_episode_csv,
                                "Avg Reward":total_reward_csv,
                                "Actor Loss" : total_actor_enegry_csv,
                                "Critic Loss": total_critic_csv,
                                "AoI":total_aoi_csv,
                                "Energy Consumption":total_energy_con_csv
                                })
    data_frame.to_csv('training_log_updated.csv',index=False,header=True,encoding='utf-8')                             
if __name__ == "__main__":
    main()

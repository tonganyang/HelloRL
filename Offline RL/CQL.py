# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:21:03 2023

conservative Q-learning，CQL

@author: TAY
"""

import random
import gym
import collections
import numpy as np
import argparse
from tqdm import tqdm
import os
import pickle
import torch
import torch.nn.functional as F
from torch.distributions import Normal #正态分布
import matplotlib.pyplot as plt
import matplot

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std) # 正态分布采样
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob

class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 策略网络
        
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  #对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 对倒立摆环境的奖励进行重塑
        
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean( F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean( F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--alg_name', default='CQL', type=str)
parser.add_argument('--path', default='E:/HFUT/RL/MountainCar/Figure', type=str)
parser.add_argument('--env_name', default='LunarLander-v2', type=str) # pip install gym[box2d]
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--tau', default=0.005, type=float) # 软更新参数
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_episodes', default=100, type=int)
parser.add_argument('--minimal_size', default=1000, type=int)
parser.add_argument('--buffer_size', default=100000, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--actor_lr', default=3e-4, type=float)
parser.add_argument('--critic_lr', default=3e-3, type=float)
parser.add_argument('--alpha_lr', default=3e-4, type=float)
args = parser.parse_known_args()[0]

def Training(env_name,
             seed,
             hidden_dim,
             actor_lr,
             critic_lr,
             gamma,
             num_episodes,
             alpha_lr,
             tau,
             buffer_size):
    
    mean = []
    std = []
    return_list = []
    
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
    env = gym.make(env_name)
    env.seed(seed=seed)
    env.reset(seed=seed)
    random.seed(seed)
    
    replay_buffer = ReplayBuffer(buffer_size)
    
    state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n 离散动作空间
    action_dim = env.action_space.shape[0] # 连续动作空间
    action_bound = env.action_space.high[0]  # 动作最大值
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    target_entropy = -env.action_space.shape[0]
    
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    
    with tqdm(total=int(num_episodes), desc='SEED %d' % seed) as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (i_episode + 1), 
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
            
            mean.append(np.mean(return_list))
            std.append(np.std(return_list))
            
    return return_list, mean, std

def save_data(path, alg_name, env_name, seed, return_list, mean, std):
    
    d = {"return_list": return_list, "mean": mean, "std": std}
    
    if not os.path.exists(path +"/" + "test_data_" + alg_name+ "_" + env_name + "_seed_" + str(seed)):
        os.makedirs(path, exist_ok=True)
        
    with open(os.path.join(path + "/" + "test_data_" + alg_name+ "_" + env_name + "_seed_" + str(seed) +".pkl"), "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def read_data(path, alg_name, env_name, seed):
    
    file = path + "/" + "test_data_" + alg_name+ "_" + env_name + "_seed_" + str(seed) +".pkl"
    
    with open(file, "rb") as f:
        data = pickle.load(f)
        return_list = data["return_list"]
        mean = data["mean"]
        std = data["std"]
        
    return return_list, mean, std

return_list, mean, std = Training(args.env_name, args.seed, args.hidden_dim, args.actor_lr, 
                                  args.critic_lr, args.gamma, args.num_episodes, 
                                  args.alpha_lr, args.tau, args.buffer_size)

save_data(args.path, args.alg_name, args.env_name, args.seed, return_list, mean, std)

return_list, mean, std = read_data(args.path, args.alg_name, args.env_name, args.seed)

matplot.MatplotlibRL(args.path, args.num_episodes, mean, std, args.alg_name, args.env_name)
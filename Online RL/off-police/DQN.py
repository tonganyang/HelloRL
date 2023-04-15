"""
DQN & DoubleDQN & Dueling DQN
"""

import random
import gym
import numpy as np
import collections
import argparse
from tqdm import tqdm
import os
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplot

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 队列，先进先出
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # random.sample() 从指定的序列中，随机的截取指定长度的片断，不作原地修改
        transitions = random.sample(self.buffer, batch_size)
        # * 表示拆分: *(1,2,3)to 1,2,3
        # zip 表示交叉合并元素: zip((1,2,3),(4,5,6),(7,8,9)) to (1,4,7),(2,5,8),(3,6,9)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
    
class DQN:
    def __init__(self, dqn_type, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, tau):
        self.action_dim = action_dim
        self.dqn_type = dqn_type
        if self.dqn_type == 'DuelingDQN':  # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
            
            self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
            
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
        self.tau = tau
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        
    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # np.random.random() 生成0~1之间的浮点数
        # np.random.random((100, 20)) 生成100行 20列的浮点数
        if np.random.random() < self.epsilon:
        # numpy.random.randint(low, high=None, size=None, dtype='l')
        # 生成一个整数或N维整数数组,取数范围: 若high不为None时,取[low,high)之间随机整数，否则取值[0,low)之间随机整数
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
            
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # a.gather(0, b) 以b中元素为索引，按a中的竖直方向进行取值
        # a.gather(1, b) 以b中元素为索引，按a中的水平方向进行取值
        q_values = self.q_net(states).gather(1, actions) # Q值
        # 下个状态的最大Q值
        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) # DQN
        
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action) # Double DQN
        
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) # 均方误差损失函数
        
        self.optimizer.zero_grad() # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward() # 反向传播更新参数
        self.optimizer.step()
        
        # 硬更新
        # if self.count % self.target_update == 0:
        #     self.target_q_net.load_state_dict(
        #         self.q_net.state_dict())  # 更新目标网络
        # self.count += 1
        
        # 软更新
        self.soft_update(self.q_net, self.target_q_net)

def Training(dqn_type,
             env_name,
             seed,
             hidden_dim,
             lr,
             batch_size,
             epsilon,
             gamma,
             target_update,
             num_episodes,
             buffer_size,
             minimal_size,
             tau):
    
    mean = []
    std = []
    return_list = []
    
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
    env = gym.make(env_name)
    # env.seed(seed=seed)
    env.reset(seed=seed)
    random.seed(seed)
    
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    agent = DQN(dqn_type, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, tau)
    
    with tqdm(total=int(num_episodes), desc='SEED %d' % seed) as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            
            state = env.reset(seed=seed)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
                    
            return_list.append(episode_return)
            
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (i_episode + 1), 
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
            
            # iters.append(i_episode)
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

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--alg_name', default='DQN(soft_update)', type=str)
parser.add_argument('--path', default='E:/HFUT/RL/MountainCar/Figure', type=str)
parser.add_argument('--env_name', default='CartPole-v0', type=str)
parser.add_argument('--tau', default=0.005, type=float) # 软更新参数
parser.add_argument('--gamma', default=0.98, type=int)
parser.add_argument('--num_episodes', default=100, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--buffer_size', default=10000, type=int)
parser.add_argument('--target_update', default=10, type=int)
parser.add_argument('--minimal_size', default=500, type=int)
parser.add_argument('--epsilon', default=0.01, type=int)
parser.add_argument('--lr', default=2e-3, type=float)
args = parser.parse_known_args()[0]

return_list, mean, std = Training(args.alg_name, args.env_name, args.seed, args.hidden_dim, args.lr, 
                                  args.batch_size, args.epsilon, args.gamma, 
                                  args.target_update, args.num_episodes, 
                                  args.buffer_size, args.minimal_size, args.tau)

# save_data(args.path, args.alg_name, args.env_name, args.seed, return_list, mean, std)

# return_list, mean, std = read_data(args.path, args.alg_name, args.env_name, args.seed)

matplot.MatplotlibRL(args.path, args.num_episodes, mean, std, args.alg_name, args.env_name)
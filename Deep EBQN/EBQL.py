import random
import gym
import numpy as np
import collections
import argparse
from tqdm import tqdm
import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplot

class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 从buffer中采样数据,数量为batch_size
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前buffer中数据的数量
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s):
        s = self.layer(s)
        return s

class Deep_EBQL:
    def __init__(self, K, hidden_dim, batch_size, lr, gamma, epsilon, target_update, num_episodes, 
                 minimal_size, env_name, seed, state_dim, action_dim, device):

        self.K = K  # 使用Qnet的个数
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.num_episodes = num_episodes
        self.minimal_size = minimal_size
        self.env = gym.make(env_name)

        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.manual_seed(seed)

        self.state_dim = state_dim #self.env.observation_space.shape[0]
        self.action_dim = action_dim  # 将连续动作分成11个离散动作

        self.device = device

        self.q_net = []
        self.optimizer = []
        
        for i in range(self.K):
            self.q_net.append(Qnet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device))
            self.optimizer.append(torch.optim.Adam(self.q_net[i].parameters(), lr=self.lr))
        self.target_q_net = copy.deepcopy(self.q_net)

    def select_action(self, state):  # epsilon-贪婪策略采取动作
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action_Q_values = torch.zeros(state.shape[0], self.action_dim)
            
            for k in range(self.K):
                action_Q_values += self.q_net[k](state)
                
            action_Q_values = action_Q_values / self.K
            action = action_Q_values.argmax().item()
            
        return action

    def max_q_value(self, state):  # 为了显示算法的过估计现象
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        for i in range(self.K):
            return self.q_net[i](state).max().item()

    def update(self, transition):
        states = torch.tensor(transition["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        ######################################################################
        kt = np.random.randint(0, self.K)
        q_values = self.q_net[kt](states).gather(1, actions)  # Q value

        max_next_q_values = torch.zeros(self.batch_size, 1).to(self.device)
        for k in range(self.K):
            if k != kt:
                max_next_q_values += self.target_q_net[k](next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        max_next_q_values = max_next_q_values / (self.K - 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD error
        ######################################################################

        loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer[kt].zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        loss.backward()  # 反向传播更新参数
        self.optimizer[kt].step()

        if self.count % self.target_update == 0:  # 更新目标网络
            for k in range(self.K):
                self.target_q_net[k].load_state_dict(self.q_net[k].state_dict())

        self.count += 1


def Training(alg_name,
             K, 
             hidden_dim, 
             batch_size, 
             lr, 
             gamma, 
             epsilon, 
             target_update, 
             num_episodes, 
             minimal_size, 
             env_name, 
             seed, 
             buffer_size):
    
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
    
    agent = Deep_EBQL(K, hidden_dim, batch_size, lr, gamma, epsilon, target_update, num_episodes, 
                      minimal_size, env_name, seed, state_dim, action_dim, device)
    
    with tqdm(total=int(num_episodes), desc='SEED %d' % seed) as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            
            state = env.reset(seed=seed)
            done = False
            while not done:
                action = agent.select_action(state)
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

parser = argparse.ArgumentParser(description='Deep EBQN parametes settings')

parser.add_argument('--alg_name', default='DEBQN(hard_update)(K=2)', type=str)
parser.add_argument('--path', default='E:/HFUT/RL/MountainCar/Figure', type=str)
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate for the net.')
parser.add_argument('--num_episodes', type=int, default=10000, help='the num of train epochs')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='Random seed.')
parser.add_argument('--gamma', type=float, default=0.98, metavar='S', help='the discount rate')
parser.add_argument('--epsilon', type=float, default=0.01, metavar='S', help='the epsilon rate')
parser.add_argument('--K', type=int, default=2, metavar='S', help='the number of Qnet used to algorithm')
parser.add_argument('--target_update', type=float, default=10, metavar='S', help='the frequency of the target net')
parser.add_argument('--buffer_size', type=float, default=10000, metavar='S', help='the size of the buffer')
parser.add_argument('--minimal_size', type=float, default=500, metavar='S', help='the minimal size of the learning')
parser.add_argument('--hidden_dim', type=float, default=128, metavar='S', help='the size of the hidden layer')
parser.add_argument('--env_name', type=str, default="CartPole-v0", metavar='S', help='the name of the environment') # Pendulum-v1

args = parser.parse_args()

return_list, mean, std = Training(args.alg_name,
                                  args.K, 
                                  args.hidden_dim, 
                                  args.batch_size, 
                                  args.lr, 
                                  args.gamma, 
                                  args.epsilon, 
                                  args.target_update, 
                                  args.num_episodes, 
                                  args.minimal_size, 
                                  args.env_name, 
                                  args.seed, 
                                  args.buffer_size)

save_data(args.path, args.alg_name, args.env_name, args.seed, return_list, mean, std)

return_list, mean, std = read_data(args.path, args.alg_name, args.env_name, args.seed)

matplot.MatplotlibRL(args.path, args.num_episodes, mean, std, args.alg_name, args.env_name)
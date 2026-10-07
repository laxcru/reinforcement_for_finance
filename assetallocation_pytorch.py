#
# Investing Environment and Agent
# Three Asset Case
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

import os
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
from pylab import plt, mpl
from scipy.optimize import minimize

import torch
from dqlagent_pytorch import *

plt.style.use('seaborn-v0_8')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True)



class observation_space:
    def __init__(self, n):
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        rn = np.random.random(3)
        return rn / rn.sum()


class Investing:
    def __init__(self, asset_one, asset_two, asset_three,
                 steps=252, amount=1):
        self.asset_one = asset_one
        self.asset_two = asset_two
        self.asset_three = asset_three
        self.steps = steps
        self.initial_balance = amount
        self.portfolio_value = amount
        self.portfolio_value_new = amount
        self.observation_space = observation_space(4)
        self.osn = self.observation_space.shape[0]
        self.action_space = action_space(3)
        self.retrieved = 0
        self._generate_data()
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _generate_data(self):
        if self.retrieved:
            pass
        else:
            url = 'https://certificate.tpq.io/rl4finance.csv'
            self.raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
            self.retrieved
        self.data = pd.DataFrame()
        self.data['X'] = self.raw[self.asset_one]
        self.data['Y'] = self.raw[self.asset_two]
        self.data['Z'] = self.raw[self.asset_three]
        s = random.randint(self.steps, len(self.data))
        self.data = self.data.iloc[s-self.steps:s]
        self.data = self.data / self.data.iloc[0]

    def _get_state(self):
        Xt = self.data['X'].iloc[self.bar]
        Yt = self.data['Y'].iloc[self.bar]
        Zt = self.data['Z'].iloc[self.bar]
        date = self.data.index[self.bar]
        return np.array(
            [Xt, Yt, Zt, self.xt, self.yt, self.zt]
            ), {'date': date}
        
    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            
    def reset(self):
        self.xt = 0
        self.yt = 0
        self.zt = 0
        self.bar = 0
        self.treward = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_value_new = self.initial_balance
        self.episode += 1
        self._generate_data()
        self.state, info = self._get_state()
        return self.state, info

    def add_results(self, pl):
        df = pd.DataFrame({
                   'e': self.episode, 'date': self.date, 
                   'xt': self.xt, 'yt': self.yt, 'zt': self.zt,
                   'pv': self.portfolio_value,
                   'pv_new': self.portfolio_value_new, 'p&l[$]': pl,
                   'p&l[%]': pl / self.portfolio_value_new * 100,
                   'Xt': self.state[0], 'Yt': self.state[1],
                   'Zt': self.state[2], 'Xt_new': self.new_state[0],
                   'Yt_new': self.new_state[1],
                   'Zt_new': self.new_state[2],
                          }, index=[0])
        self.portfolios = pd.concat((self.portfolios, df), ignore_index=True)
        
    def step(self, action):
        self.bar += 1
        self.new_state, info = self._get_state()
        self.date = info['date']
        if self.bar == 1:
            self.xt = action[0]
            self.yt = action[1]
            self.zt = action[2]
            pl = 0.
            reward = 0.
            self.add_results(pl)
        else:
            self.portfolio_value_new = (
                self.xt * self.portfolio_value *
                    self.new_state[0] / self.state[0] +
                self.yt * self.portfolio_value *
                    self.new_state[1] / self.state[1] +
                self.zt * self.portfolio_value *
                    self.new_state[2] / self.state[2]
            )
            pl = self.portfolio_value_new - self.portfolio_value
            self.xt = action[0]
            self.yt = action[1]
            self.zt = action[2]
            self.add_results(pl)
            ret = self.portfolios['p&l[%]'].iloc[-1] / 100 * 252
            vol = self.portfolios['p&l[%]'].rolling(
                20, min_periods=1).std().iloc[-1] * math.sqrt(252)
            sharpe = ret / vol
            reward = sharpe
            self.portfolio_value = self.portfolio_value_new
        if self.bar == len(self.data) - 1:
            done = True
        else:
            done = False
        self.state = self.new_state
        return self.state, reward, done, False, {}
        

class InvestingAgent(DQLAgent):
    def __init__(self, symbol, feature, n_features, env, hu=24, lr=0.001):
        super().__init__(symbol, feature, n_features, env, hu, lr)
        # Continuous action: override model to output scalar Q-value
        self.model = QNetwork(self.n_features, 1, hu).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    def opt_action(self, state):
        bnds = 3 * [(0, 1)]  # three weights
        cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1}]
        def f_obj(x):
            s = state.copy()
            s[0, 3] = x[0]
            s[0, 4] = x[1]
            s[0, 5] = x[2]
            pen = np.mean((state[0, 3:] - x) ** 2)
            s_tensor = torch.FloatTensor(s).to(device)
            with torch.no_grad():
                q_val = self.model(s_tensor)
            return q_val.cpu().numpy()[0, 0] - pen
        try:
            state = self._reshape(state)
            res = minimize(lambda x: -f_obj(x), 3 * [1 / 3],
                           bounds=bnds, constraints=cons,
                           options={'eps': 1e-4}, method='SLSQP')
            action = res['x']
        except Exception:
            action = self.env.action_space.sample()
        return action
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return self.opt_action(state)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            target = torch.tensor([reward], dtype=torch.float32).to(device)
            if not done:
                ns = next_state.copy()
                action_cont = self.opt_action(ns)
                ns[0, 3:] = action_cont
                ns_tensor = torch.FloatTensor(ns).to(device)
                with torch.no_grad():
                    future_q = self.model(ns_tensor)[0, 0]
                target = target + self.gamma * future_q
            state_tensor = torch.FloatTensor(state).to(device)
            self.optimizer.zero_grad()
            current_q = self.model(state_tensor)[0, 0]
            loss = self.criterion(current_q, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def test(self, episodes, verbose=True):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for _ in range(1, len(self.env.data) + 1):
                action = self.opt_action(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                treward += reward
                if done:
                    templ = f'episode={e} | total reward={treward:4.2f}'
                    if verbose:
                        print(templ, end='\r')
                    break
        print()


#
# Deep Q-Learning (DQL) Agent with PyTorch
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#
# 功能概述：
#   实现一个基于深度Q学习（Deep Q-Learning）算法的强化学习智能体。
#   DQL 是一种 off-policy 的离策略学习方法，使用神经网络来近似 Q 值函数，
#   解决传统 Q-Learning 在状态空间过大时无法维护 Q 表的问题。
#
#   DQL 的三大核心技术：
#   1. 经验回放（Experience Replay）：将每一步的 (s, a, r, s', done) 存入回放缓冲区，
#      训练时随机采样一个 mini-batch 来更新网络，打破样本间的相关性
#   2. Epsilon-Greedy 探索策略：以 epsilon 概率随机探索，以 1-epsilon 概率利用已知最优动作，
#      训练初期 epsilon 大（多探索），训练后期 epsilon 衰减（多利用）
#   3. Q-Learning 目标：用贝尔曼方程的目标值 target = r + gamma * max_a' Q(s', a')
#      作为监督信号，用均方误差损失来训练网络
#
#   智能体与环境（如 Finance、Simulation）通过 Gym 风格的接口交互：
#     state, info = env.reset()          # 重置环境，获取初始状态
#     next_state, reward, done, ... = env.step(action)  # 执行动作，获取反馈
#

import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# 忽略警告信息，保持输出整洁
warnings.simplefilter('ignore')

# 固定 PYTHONHASHSEED 以提高实验可复现性
os.environ['PYTHONHASHSEED'] = '0'

# 自动检测 GPU，如果有则用 GPU 加速，否则用 CPU
# DQL 网络较小，CPU 即可胜任，但 GPU 在批量训练时会更快
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):
    """
    Q 值网络——用前馈神经网络近似 Q 函数。

    原理：
      在传统 Q-Learning 中，Q(s, a) 以 Q 表（二维数组）形式存储，
      但当状态空间巨大（比如连续值、高维向量）时，Q 表根本存不下。
      DQL 用一个神经网络 Q(s, a; theta) 来近似真实的 Q 值函数，
      其中 theta 是网络的权重参数，通过训练不断更新。

    网络结构：
      三层全连接网络（MLP），两层隐藏层 + 一层输出层：
        输入层 → 隐藏层(Linear + ReLU) → 隐藏层(Linear + ReLU) → 输出层(Linear)
      - 激活函数用 ReLU：相比 sigmoid/tanh，ReLU 能缓解梯度消失问题，训练更快
      - 输出层不激活：Q 值可以是任意实数（正或负），所以不需要激活函数

    Attributes:
      state_dim (int): 输入维度 = 状态特征数量（n_features）
      action_dim (int): 输出维度 = 动作空间大小（env.action_space.n）
      hu (int): 每个隐藏层的神经元数量（hidden units）
    """

    def __init__(self, state_dim, action_dim, hu=24):
        """
        初始化 Q 网络。

        Args:
            state_dim (int): 输入维度，等于环境的状态特征数
            action_dim (int): 输出维度，等于环境的动作空间大小
            hu (int, optional): 隐藏层神经元数量。默认 24。
                这里用的是一个很小的网络（24 个神经元），
                因为金融环境的状态空间本身就不大（n_features 通常 4~8），
                小网络既能学到东西，又不容易过拟合。
        """
        super(QNetwork, self).__init__()
        # 第一层：输入 state_dim → 隐藏层 hu
        self.fc1 = nn.Linear(state_dim, hu)
        # 第二层：隐藏层 hu → 隐藏层 hu
        self.fc2 = nn.Linear(hu, hu)
        # 输出层：隐藏层 hu → 动作数 action_dim
        # 输出的是每个动作对应的 Q 值 Q(s, a_i)
        self.fc3 = nn.Linear(hu, action_dim)

    def forward(self, x):
        """
        前向传播：输入状态 x，输出每个动作的 Q 值。

        Args:
            x (torch.Tensor): 输入状态张量，形状 (batch_size, state_dim)

        Returns:
            torch.Tensor: 每个动作的 Q 值，形状 (batch_size, action_dim)
                例如 action_dim=2 时，输出 [Q(s, a=0), Q(s, a=1)]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQLAgent:
    """
    Deep Q-Learning 智能体——完整的 DQL 训练和测试框架。

    训练流程（learn 方法）：
      for episode in range(episodes):
          state = env.reset()              # 重置环境
          while not done:
              action = epsilon_greedy(state)  # 选动作
              next_state, reward, done = env.step(action)  # 执行
              memory.append(state, action, reward, next_state, done)  # 存经验
              state = next_state
          replay()   # 一个 episode 结束后，从经验池采样训练网络

    核心算法参数：
      epsilon：探索率，控制随机探索的概率。初始 1.0（全随机），
               每步乘以 epsilon_decay 衰减到 epsilon_min（0.1）
      gamma：折扣因子，衡量未来奖励的重要性。gamma 越小越关注短期奖励，
             gamma 越大会考虑长期累计奖励。这里 gamma=0.5 是一个较小值，
             因为金融预测中"今天的方向"比"10天后的方向"更重要
      memory：经验回放缓冲区，用 deque 实现，最多存 2000 条经验，
              超出时自动丢弃最早的（FIFO）
      batch_size：每次从 memory 采样多少条经验来训练网络

    Attributes:
      epsilon (float): 当前探索率
      epsilon_decay (float): 每步的衰减系数（0.9975 意味着大约 1850 步后减半）
      epsilon_min (float): 探索率的下限，避免完全停止探索
      memory (deque): 经验回放缓冲区，存储 (s, a, r, s', done) 五元组
      batch_size (int): replay 时的 mini-batch 大小
      gamma (float): 折扣因子
      trewards (list): 每个训练 episode 的累计奖励记录
      max_treward (float): 训练过程中见过的最高累计奖励
      n_features (int): 状态特征数量
      env: 环境对象（Finance 或 Simulation 等）
      episodes (int): 已经训练过的 episode 数
      model (QNetwork): Q 值网络
      optimizer: Adam 优化器
      criterion: 均方误差损失函数（MSELoss）
      performances (list, optional): 只有测试且环境有 min_performance 属性时才会填充，
          存储每个测试 episode 的最终净值（用于绘制 performances 直方图）
    """

    def __init__(self, symbol, feature, n_features, env, hu=24, lr=0.001):
        """
        初始化 DQL 智能体。

        Args:
            symbol (str): 交易/预测的标的名称（如 'GLD'）
                注意：这个参数在当前代码里没有实际使用，只是记录一下
            feature (str): 观察特征的列名（如 'r'）
                注意：这个参数也没有实际使用
            n_features (int): 状态特征数量，即每次观察的窗口长度
            env: 强化学习环境对象，需要实现 reset() 和 step() 方法，
                 且有 action_space.n 属性
            hu (int, optional): Q 网络隐藏层神经元数。默认 24
            lr (float, optional): Adam 优化器的学习率。默认 0.001
        """
        # ===== Epsilon-Greedy 探索策略参数 =====
        # 初始探索率 1.0 → 训练初期完全随机探索，不利用任何已有知识
        self.epsilon = 1.0
        # 衰减系数 0.9975 → 每训练一步，epsilon 乘以此系数
        # 衰减速度设计：约 1850 步后 epsilon 从 1.0 降到 0.5
        # 约 9200 步后降到 0.1（到达下限）
        self.epsilon_decay = 0.9975
        # 探索率下限 → 即使训练很久也保持 10% 的随机探索，
        # 防止策略过早收敛到局部最优
        self.epsilon_min = 0.1

        # ===== 经验回放缓冲区 =====
        # deque(maxlen=2000)：最多存 2000 条经验，新的进来自动挤掉最旧的
        # FIFO 机制保证经验分布不会无限偏向后期的数据
        self.memory = deque(maxlen=2000)
        # 每次从 memory 采样 32 条经验来训练网络
        self.batch_size = 32

        # ===== Q-Learning 参数 =====
        # 折扣因子 gamma → target = r + gamma * max Q(s', a')
        # gamma=0.5 是较小值，表示"只关心当前步和下一步，不太遥远的未来不重要"
        # 这对金融预测是合理的：今天预测对了就有奖励，太远的未来受太多因素影响
        self.gamma = 0.5

        # ===== 训练记录 =====
        # 每个训练 episode 的累计奖励，用于观察训练趋势
        self.trewards = []
        # 训练过程中见过的最高累计奖励，用于监控是否在进步
        self.max_treward = -np.inf

        # ===== 与环境的连接 =====
        self.n_features = n_features
        self.env = env
        self.episodes = 0

        # ===== Q 网络 + 优化器 + 损失函数 =====
        # 网络输入 = n_features（状态维度），输出 = env.action_space.n（动作数）
        # 网络搬到 device（GPU 或 CPU）上
        self.model = QNetwork(self.n_features,
                              self.env.action_space.n, hu).to(device)
        # Adam 优化器：自适应学习率，比 SGD 收敛更快更稳定
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 均方误差损失：让网络的 Q 值逼近 Q-Learning 的目标值
        self.criterion = nn.MSELoss()

    def _reshape(self, state):
        """
        状态预处理：将环境返回的一维状态 reshape 成 (1, state_dim) 的二维数组。

        为什么需要这一步？
          Q 网络的 forward 期望输入形状是 (batch_size, state_dim)，
          但环境 reset()/step() 返回的 state 是一维数组 (state_dim,)。
          所以需要加一个 batch 维度，把 (n_features,) 变成 (1, n_features)。

        Args:
            state (np.ndarray): 环境返回的原始状态，形状 (n_features,)

        Returns:
            np.ndarray: reshape 后的状态，形状 (1, n_features)
        """
        state = state.flatten()
        return np.reshape(state, [1, len(state)])

    def act(self, state):
        """
        根据当前状态选择动作——Epsilon-Greedy 策略。

        策略逻辑：
          以 epsilon 概率 → 随机选一个动作（探索 exploration）
          以 1-epsilon 概率 → 选 Q 值最大的动作（利用 exploitation）

        训练初期 epsilon ≈ 1.0 → 几乎全在探索
        训练后期 epsilon → 0.1 → 90% 利用已学到的知识，10% 继续探索

        Args:
            state (np.ndarray): 当前状态，形状 (1, n_features)
                （注意：这里期望是已经过 _reshape 的二维数组）

        Returns:
            int: 选择的动作索引，0 或 1（取决于 env.action_space.n）
        """
        # ===== 探索：以 epsilon 概率随机选 =====
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        # ===== 利用：神经网络预测各动作 Q 值，选最大的 =====
        # 转为 PyTorch 张量并搬到 device 上
        state_tensor = torch.FloatTensor(state).to(device)

        # 如果输入是一维的（比如调用方忘了 _reshape），补一个 batch 维度
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # torch.no_grad()：推理模式，不计算梯度、不跟踪计算图
        # 因为这里只是选动作，不需要反向传播，no_grad 能节省显存和计算
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # argmax 选 Q 值最大的动作，转为 Python int 返回
        # q_values 形状 (1, action_dim)，q_values[0] 取第一个（也是唯一的）样本
        return int(torch.argmax(q_values[0]).item())

    def replay(self):
        """
        经验回放训练——从 memory 中随机采样一个 mini-batch，
        用 Q-Learning 的目标值来更新 Q 网络。

        这是 DQL 算法的核心训练步骤，每次调用做以下事情：

        1. 从 memory 随机采样 batch_size 条经验
           为什么要"随机采样"？——打破经验之间的时间相关性，
           让训练数据更接近独立同分布（i.i.d.），稳定训练
        2. 计算当前 Q 值：Q(s, a)，即网络对 (state, action) 对的预测
        3. 计算 Q-Learning 目标值：
              target = r + gamma * max_a' Q(s', a')     （如果 done=False）
              target = r                                （如果 done=True，因为没有下一状态了）
           为什么用 detach()？——target 是"监督信号"，不应该参与梯度反向传播，
           否则网络会 chasing its own tail（追着自己的预测跑）
        4. 计算 loss = MSE(current_q, target)，反向传播更新网络

        epsilon 衰减也在这里执行（每次 replay 衰减一次）

        注意：只有当 memory 中的经验数量 >= batch_size 时才会执行，
        否则直接返回（没法采样）。
        """
        # 经验池太小，采样不了一个 batch → 跳过，等攒够了再说
        if len(self.memory) < self.batch_size:
            return

        # 从经验池随机采样 batch_size 条（不放回抽样）
        batch = random.sample(self.memory, self.batch_size)

        # 将 batch 拆解为独立的数组，方便批量处理
        # 每条经验的格式：(state, action, next_state, reward, done)
        states = np.vstack([e[0] for e in batch])       # (batch, state_dim)
        actions = np.array([e[1] for e in batch])        # (batch,)
        next_states = np.vstack([e[2] for e in batch])   # (batch, state_dim)
        rewards = np.array([e[3] for e in batch], dtype=np.float32)  # (batch,)
        dones = np.array([e[4] for e in batch], dtype=bool)           # (batch,)

        # 将 numpy 数组转为 PyTorch 张量，搬到 device 上
        states_tensor = torch.FloatTensor(states).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        # actions 需要是 LongTensor 且 shape (batch, 1)，因为 gather 需要索引是 long 类型
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)

        # ===== Step 1：计算当前 Q 值 Q(s, a) =====
        # 网络输出 (batch, action_dim)，gather 选出每个样本实际执行的那个动作的 Q 值
        # 比如 batch 中第 i 个样本执行的是 action=1，就取网络对第 i 个样本预测的 Q(s, a=1)
        current_q = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)

        # ===== Step 2：计算目标 Q 值 target = r + gamma * max Q(s', a') =====
        # next_q = max_a' Q(s', a')，即下一状态所有可能动作中最高的 Q 值
        # .max(1)[0] 返回 dim=1（动作维度）上的最大值，[0] 取 values（不取 indices）
        next_q = self.model(next_states_tensor).max(1)[0]

        # Q-Learning 贝尔曼目标：
        #   done=False → target = r + gamma * next_q  （有下一状态）
        #   done=True  → target = r                   （episode 结束，没有下一状态）
        # (~dones_tensor).float() 把 bool 取反后转 float（True→1.0, False→0.0）
        # 这样 done=True 的样本的 gamma * next_q 部分就变成 0 了
        target_q = rewards_tensor + self.gamma * next_q * (~dones_tensor).float()

        # ===== Step 3：计算损失并反向传播 =====
        # target_q.detach()：阻断梯度传播，target 作为固定的监督信号
        # 如果不 detach，网络更新时 target_q 也会跟着变，导致训练不稳定
        loss = self.criterion(current_q, target_q.detach())
        # 梯度清零（PyTorch 默认累加梯度，每次反向传播前必须手动清零）
        self.optimizer.zero_grad()
        # 反向传播：计算梯度
        loss.backward()
        # 更新网络参数
        self.optimizer.step()

        # ===== Step 4：Epsilon 衰减 =====
        # 每次 replay 后 epsilon 乘以衰减系数，但不低于 epsilon_min
        # 随着训练进行，智能体会越来越倾向于利用已学到的知识
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        """
        训练 DQL 智能体——在环境上跑指定数量的 episode。

        训练流程：
          for episode in range(episodes):
              state = env.reset()
              treward = 0
              for step in range(max_steps):
                  action = self.act(state)          # Epsilon-Greedy 选动作
                  next_state, reward, done = env.step(action)  # 环境执行
                  treward += reward
                  memory.append(state, action, ...)  # 存经验到回放池
                  state = next_state
                  if done:
                      break
              if memory 够大:
                  replay()    # 训练网络（从经验池采样）

        关键设计：
          - 每个 episode 最多 5000 步（for f in range(1, 5000)），防止死循环
          - 每个 episode 结束后只 replay 一次（不是每步都 replay），
            这是为了稳定训练——DQL 论文里也是这么建议的
          - 如果环境的 reset() 会重新生成数据（如 Simulation.new=True），
            那么每次 episode 实际上是在不同的随机路径上训练，
            这自动实现了"域随机化"，防止过拟合

        Args:
            episodes (int): 要训练的 episode 总数
        """
        for e in range(1, episodes + 1):
            self.episodes += 1

            # 重置环境，获取初始状态
            # Simulation/Trading 类 new=True 时会在这里重新生成随机数据
            state, _ = self.env.reset()
            state = self._reshape(state)

            # 本 episode 的累计奖励
            treward = 0

            # 每个 episode 最多跑 5000 步，防止无限循环
            for f in range(1, 5000):
                self.f = f
                # Epsilon-Greedy 选动作
                action = self.act(state)
                # 环境执行动作，返回反馈
                next_state, reward, done, trunc, _ = self.env.step(action)
                # 累计奖励
                treward += reward
                next_state = self._reshape(next_state)
                # 将这条经验存入回放缓冲区
                self.memory.append((state, action, next_state, reward, done))
                # 状态推进
                state = next_state
                if done:
                    # episode 结束，记录累计奖励
                    self.trewards.append(treward)
                    # 更新历史最佳累计奖励
                    self.max_treward = max(self.max_treward, treward)
                    # 打印进度，\r 覆盖同一行，避免刷屏
                    templ = f'episode={self.episodes:4d} | '
                    templ += f'treward={treward:7.3f} | max={self.max_treward:7.3f}'
                    print(templ, end='\r')
                    break

            # 每个 episode 结束后，如果经验池够大，就 replay 一次训练网络
            # 注意：这里是"每个 episode replay 一次"，而不是"每步 replay 一次"
            # 这样做的好处是训练更稳定，且和环境交互的开销更小
            if len(self.memory) > self.batch_size:
                self.replay()
        print()

    def test(self, episodes, min_accuracy=0.0, min_performance=0.0,
             verbose=True, full=True):
        """
        测试 DQL 智能体——用训练好的策略在环境上跑指定数量的 episode，
        评估策略的表现。

        与 learn() 的区别：
          - 测试时 epsilon 不衰减（但 act() 里的 epsilon 仍然生效，
            所以测试时有少量随机探索。如果想完全利用，可以临时设 epsilon=0）
          - 测试时不调用 replay()，网络参数固定不变
          - 测试时临时放宽环境的终止阈值（min_accuracy=0, min_performance=0），
            让每个 episode 都能完整跑完，公平评估最终表现
          - 如果环境有 min_performance 属性（如 Notebook 手写 Trading 类），
            会额外记录每个 episode 的最终净值到 self.performances，
            供后续绘制直方图使用

        关键设计——阈值的"备份-临时修改-恢复"：
          训练时环境的 min_accuracy/min_performance 可能有较高的值
          （如 min_accuracy=0.5, min_performance=0.85），
          让表现差的 episode 提前终止。但测试时要公平评估，
          所以临时设为 0，跑完后再恢复原值，不影响后续训练。

        Args:
            episodes (int): 要测试的 episode 数
            min_accuracy (float, optional): 测试时临时设置的准确率阈值。
                默认 0.0，即测试时不做准确率提前终止
            min_performance (float, optional): 测试时临时设置的净值阈值。
                默认 0.0，即测试时不做净值提前终止
            verbose (bool, optional): 是否打印每个 episode 的结果。默认 True
            full (bool, optional): True 时每个 episode 换行打印，
                False 时同一行覆盖打印（和 learn 一样的风格）。默认 True
        """
        # ===== 备份环境的原始阈值，然后临时放宽 =====
        # 这样测试时不会因为准确率太低或净值跌破阈值而提前终止 episode
        ma = getattr(self.env, 'min_accuracy', None)
        if hasattr(self.env, 'min_accuracy'):
            self.env.min_accuracy = min_accuracy
        mp = None
        if hasattr(self.env, 'min_performance'):
            mp = self.env.min_performance
            self.env.min_performance = min_performance
            # 如果环境支持净值追踪，初始化 performances 列表
            # 这是给 Notebook 里绘制 performances 直方图用的
            self.performances = []

        # ===== 跑 episodes 个测试 episode =====
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)

            # 最多 5000 步（和 learn 一样的上限）
            for f in range(1, 5001):
                # 选动作（注意：这里仍然用 act()，所以 epsilon-greedy 仍然生效，
                # 如果测试时想完全利用，需要外部先设 agent.epsilon = 0）
                action = self.act(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                if done:
                    # 打印本 episode 的结果：用了多少步（f）、准确率、可选的净值
                    templ = f'total reward={f:4d} | accuracy={self.env.accuracy:.3f}'
                    if hasattr(self.env, 'min_performance'):
                        # 只有环境有 performance 属性时才记录
                        # Notebook 手写 Trading 类有，Finance 类没有
                        self.performances.append(self.env.performance)
                        templ += f' | performance={self.env.performance:.3f}'
                    if verbose:
                        if full:
                            print(templ)       # 每个 episode 换行
                        else:
                            print(templ, end='\r')  # 同一行覆盖
                    break

        # ===== 恢复环境的原始阈值 =====
        # 保证测试后环境状态不变，不影响后续可能的训练或再测试
        if hasattr(self.env, 'min_accuracy') and ma is not None:
            self.env.min_accuracy = ma
        if mp is not None:
            self.env.min_performance = mp
        print()
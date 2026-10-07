#
# Monte Carlo Simulation Environment
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#
# 功能概述：
#   实现一个基于 Ornstein-Uhlenbeck（OU）均值回复过程的蒙特卡罗模拟环境。
#   与 Finance 类使用真实历史数据不同，本环境可以无限生成随机价格序列，
#   用于强化学习智能体的训练和测试。
#
#   OU 过程是金融中经典的均值回复模型，SDE 形式为：
#     dS_t = κ(θ - S_t)dt + σ S_t dW_t
#   其中：
#     κ (kappa) —— 均值回复速度，越大越快往长期均值 θ 回归
#     θ (theta) —— 长期均值水平
#     σ (sigma) —— 波动率
#   适合模拟配对价差、统计套利标的等具有均值回复特征的资产。
#
#   与 Finance 类的核心区别：
#     1. 数据来源：OU 随机模拟 vs 真实历史 CSV
#     2. reset() 行为：new=True 时每次重新生成数据 vs 固定数据
#     3. 用途：域随机化训练/泛化测试 vs 真实数据验证
#

import math
import random
import numpy as np
import pandas as pd
from numpy.random import default_rng
import torch

# NumPy 新版随机数生成器（PCG64 算法，比旧版 MT19937 更快）
rng = default_rng()


class ActionSpace:
    """
    动作空间类——定义智能体可以执行的所有动作。

    本环境与 Finance 类一样是"涨跌方向预测"任务，
    因此只有 2 个动作：
      action = 0  → 预测当前 bar 下跌
      action = 1  → 预测当前 bar 上涨
    """

    # 动作总数：0 和 1 两种选择
    n = 2

    def sample(self):
        """
        从动作空间中随机采样一个动作。
        用于 epsilon-greedy 探索策略中的随机选择。

        Returns:
            int: 0 或 1，代表随机选择的预测方向
        """
        return random.randint(0, 1)


class Simulation:
    """
    蒙特卡罗模拟强化学习环境——基于 OU 过程生成随机价格序列。

    数据生成流程：
      _simulate_data()  → 用 OU 过程生成一条价格路径
      _prepare_data()   → 计算对数收益率、生成涨跌标记、标准化特征

    reset() 的两种行为（由 new 参数控制）：
      new=False → reset() 不重新生成数据，重复使用同一条路径
      new=True  → reset() 每次都重新生成一条全新的 OU 路径

    交互流程（Gym 风格）：
      reset() → 重置环境（可能换数据），返回初始状态
      step(action) → 执行预测动作，返回 (下一状态, 奖励, 是否结束, 截断标志, 额外信息)

    Attributes:
      symbol (str): 标的标签（纯标识符，不对应真实资产）
      feature (str): 用作观察特征的列名，如 'r' 表示用收益率
      n_features (int): 观察窗口长度（智能体每次看多少根历史K线）
      start (str): 模拟时间区间起点，如 '2024-1-1'
      end (str): 模拟时间区间终点，如 '2026-1-1'
      periods (int): 模拟数据点总数
      x0 (float): OU 过程的初始价格水平
      kappa (float): OU 过程的均值回复速度
      theta (float): OU 过程的长期均值
      sigma (float): OU 过程的波动率
      min_accuracy (float): 提前终止的准确率阈值
      normalize (bool): 是否对特征做 Z-score 标准化
      new (bool): reset() 时是否重新生成数据
      action_space (ActionSpace): 动作空间对象
      data (pd.DataFrame): 原始模拟数据，包含价格、收益率、涨跌标记
      data_ (pd.DataFrame): 标准化后的数据，作为智能体的观察输入
      bar (int): 当前进度指针
      treward (int): 累计正确预测次数
      accuracy (float): 当前预测准确率
      mu (pd.Series): 各列的均值（仅 normalize=True 时存在）
      std (pd.Series): 各列的标准差（仅 normalize=True 时存在）
    """

    def __init__(self, symbol, feature, n_features,
                 start, end, periods,
                 min_accuracy=0.525, x0=100,
                 kappa=1, theta=100, sigma=0.2,
                 normalize=True, new=False):
        """
        初始化模拟环境。

        Args:
            symbol (str): 标的标签，如 'SYMBOL'、'GLD_SIM'
                仅作标识用，不会去查找真实资产
            feature (str): 观察特征列名，如 'r'（收益率）或 symbol（价格本身）
            n_features (int): 观察窗口长度，即每次 state 包含多少根历史K线
            start (str): 模拟时间区间起点，如 '2024-1-1'
            end (str): 模拟时间区间终点，如 '2026-1-1'
            periods (int): 模拟数据的点数，如 2*252 表示约 2 年日频
            min_accuracy (float, optional): 提前终止的最低准确率阈值。
                默认 0.525——比 Finance 类的 0.485 稍高，
                因为模拟数据是可控的 OU 过程，理论上更容易预测。
            x0 (float, optional): OU 初始价格。默认 100
            kappa (float, optional): 均值回复速度 κ。默认 1
            theta (float, optional): 长期均值 θ。默认 100
            sigma (float, optional): 波动率 σ。默认 0.2（20%）
            normalize (bool, optional): 是否做 Z-score 标准化。默认 True
            new (bool, optional): reset() 时是否重新生成数据。默认 False。
                True → 每次 reset 生成新的 OU 路径（域随机化训练用）
                False → 固定同一条路径（测试或调试用）
        """
        self.symbol = symbol
        self.feature = feature
        self.n_features = n_features
        self.start = start
        self.end = end
        self.periods = periods

        # OU 过程参数
        self.x0 = x0          # 初始价格
        self.kappa = kappa    # 均值回复速度 κ
        self.theta = theta    # 长期均值 θ
        self.sigma = sigma    # 波动率 σ

        self.min_accuracy = min_accuracy
        self.normalize = normalize
        self.new = new
        self.action_space = ActionSpace()

        # 初始化时立即生成第一条数据并做预处理
        # 后续 reset() 是否重生成取决于 self.new
        self._simulate_data()
        self._prepare_data()

    def _simulate_data(self):
        """
        用 Ornstein-Uhlenbeck 过程生成一条模拟价格路径。

        离散化 OU 过程的 SDE：
          S_t = S_{t-1} + κ(θ - S_{t-1})dt + σ S_{t-1} √dt · ε_t
        其中 ε_t ~ N(0,1) 是标准正态噪声。

        dt 的计算：
          模拟区间的总天数除以 365 换算成年，再除以 periods 得到每步的时间间隔。
          这样 sigma 的单位就是"年化波动率"，与金融惯例一致。

        生成的路径具有以下统计特征：
          - 长期会在 θ 附近波动（均值回复）
          - 波动率与当前价格水平 S_{t-1} 成正比（几何布朗运动风格）
          - 增量之间有负自相关（偏离均值后会拉回来）
        """
        # 生成时间索引：从 start 到 end，均匀分布 periods 个时间点
        index = pd.date_range(start=self.start,
                              end=self.end, periods=self.periods)

        # 价格序列初始化：第一个点是初始价格 x0
        s = [self.x0]

        # 计算每步的时间间隔 dt（以年为单位）
        # (index[-1] - index[0]).days 是总天数，/365 转成年，/periods 得单步间隔
        dt = (index[-1] - index[0]).days / 365 / self.periods

        # 逐步递推生成 OU 路径
        for t in range(1, len(index)):
            # OU 离散化公式：
            # 漂移项：κ(θ - S_{t-1})dt → 往均值 θ 拉，偏离越大拉力越强
            # 随机项：σ S_{t-1} √dt · ε → 波动率与当前价格成正比，ε 是高斯噪声
            s_ = (s[t - 1]
                  + self.kappa * (self.theta - s[t - 1]) * dt
                  + s[t - 1] * self.sigma * math.sqrt(dt)
                  * random.gauss(0, 1))
            s.append(s_)

        # 存为 DataFrame，时间戳作索引，列名为 symbol
        self.data = pd.DataFrame(s, columns=[self.symbol], index=index)

    def _prepare_data(self):
        """
        对模拟价格数据做预处理，构建环境所需的特征。

        处理步骤：
          1. 计算对数收益率 r = ln(S_t / S_{t-1})
             对数收益率具有可加性（多期 = 单期之和）、更接近正态
          2. 删除第一行的 NaN（没有前一天数据无法算收益率）
          3. 可选的 Z-score 标准化：(x - mean) / std
             - normalize=True：同时保存 mu、std 供后续参考，data_ 存标准化结果
             - normalize=False：data_ 直接复制原始数据
          4. 生成涨跌标记 d：r > 0 → 1（涨），r ≤ 0 → 0（跌）
             这就是智能体需要预测的 ground truth
        """
        # 对数收益率：r_t = ln(S_t / S_{t-1})
        self.data['r'] = np.log(self.data / self.data.shift(1))

        # 删除第一行 NaN
        self.data.dropna(inplace=True)

        if self.normalize:
            # Z-score 标准化：将各列缩放到均值0、标准差1
            # 标准化后的数据喂给神经网络，能加速收敛、防止梯度爆炸
            self.mu = self.data.mean()
            self.std = self.data.std()
            self.data_ = (self.data - self.mu) / self.std
        else:
            # 不标准化，直接复制原始数据
            self.data_ = self.data.copy()

        # 生成二分类标签：1=上涨，0=下跌
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        # 显式转为 int 类型，避免后续比较时出现类型问题
        self.data['d'] = self.data['d'].astype(int)

    def _get_state(self):
        """
        从标准化数据中截取当前时刻的观察窗口。

        窗口范围：[bar - n_features, bar)，即当前 bar 之前的 n_features 根K线。
        智能体需要基于这些历史特征，对当前 bar 的涨跌做预测。

        Returns:
            pd.Series: 观察窗口对应的特征序列，长度为 n_features
        """
        return self.data_[self.feature].iloc[self.bar -
                                self.n_features:self.bar]

    def seed(self, seed):
        """
        设置随机种子，使模拟结果可复现。

        注意：此方法里有一个 tf.random.set_random_seed 调用，
        看起来是从 TensorFlow 版本复制过来时遗留的——当前项目用的是 PyTorch，
        没有 tf 依赖。实际运行时这行可能会报错（如果没有安装 tf）。

        Args:
            seed (int): 随机种子值
        """
        random.seed(seed)
        np.random.seed(seed)
        # TODO: 这行是 TensorFlow 遗留，PyTorch 项目中应改为 torch.manual_seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        """
        重置环境到初始状态，开始一个新的 episode。

        关键行为取决于 self.new：
          new=True  → 先调用 _simulate_data() + _prepare_data() 生成全新的 OU 路径
                       这是"域随机化"训练模式——每次 episode 都在不同数据上学习
          new=False → 不重新生成数据，重复使用初始化时的同一条路径
                       这是固定数据模式——用于测试或调试

        之后统一做：
          treward 归零、accuracy 归零、bar 置为 n_features、构造初始观察窗口

        Returns:
            tuple: (state, info)
              - state (np.ndarray): 初始观察，形状 (n_features,) 的一维数组
              - info (dict): 辅助信息字典，本环境返回空字典
        """
        if self.new:
            # 每次 reset 都生成全新的 OU 随机路径 + 重新预处理
            # 这是域随机化的核心：让智能体在不同数据上反复学习，防止过拟合
            self._simulate_data()
            self._prepare_data()

        # 重置累计奖励和准确率
        self.treward = 0
        self.accuracy = 0

        # bar 从 n_features 开始：前面 n_features 根K线用来构建观察窗口
        self.bar = self.n_features

        # 构造初始观察
        state = self._get_state()

        # .values 把 pd.Series 转成 np.ndarray，符合 Gym 接口习惯
        return state.values, {}

    def step(self, action):
        """
        执行一个预测动作，推进环境一个时间步。

        交互逻辑：
          1. 检查 action（智能体的预测）与 self.data['d']（真实涨跌）是否一致
             - 正确 → reward = 1
             - 错误 → reward = 0
          2. 更新累计奖励、推进 bar 指针、计算当前准确率
          3. 判断 episode 是否结束（三种终止条件，优先级见下方）
          4. 返回下一状态

        终止条件（满足任一即 done=True）：
          ① bar 走完了整条数据 → 正常结束
          ② 刚猜对（reward == 1）→ 继续，不触发提前终止
             （这是一个有趣的设计：只要还能猜对就不给提前判死刑）
          ③ 准确率低于 min_accuracy 且已跑了至少 n_features + 15 步
             → 提前终止，避免在表现太差的轨迹上浪费资源

        与 Finance.step() 几乎完全一样，唯一的小差异：
          条件③的最小步数是 self.bar > self.n_features + 15（本文件）
          vs 条件③是 self.bar > 15（finance.py）
          本质相同，只是写法不同。

        Args:
            action (int): 智能体的预测动作，0 或 1

        Returns:
            tuple: (next_state, reward, done, truncated, info)
              - next_state (np.ndarray): 下一时刻的观察，形状 (n_features,)
              - reward (int): 本步奖励，1=预测正确，0=预测错误
              - done (bool): episode 是否结束
              - truncated (bool): Gym 标准截断标志，本环境固定 False
              - info (dict): 辅助信息，本环境返回空字典
        """
        # 判断预测是否正确：action 是智能体的预测，
        # self.data['d'].iloc[self.bar] 是真实的涨跌标记（ground truth）
        if action == self.data['d'].iloc[self.bar]:
            correct = True
        else:
            correct = False

        # 二值奖励：猜对=1，猜错=0
        # 这是纯准确率奖励，不直接用 P&L，好处是避免 agent 冒大险
        reward = 1 if correct else 0

        # 更新累计奖励和进度
        self.treward += reward
        self.bar += 1

        # 当前准确率 = 累计正确次数 / 已预测步数
        # 分母是 (bar - n_features)，因为前 n_features 步是用来构建初始窗口的
        self.accuracy = self.treward / (self.bar - self.n_features)

        # ===== 判断 episode 是否结束 =====
        if self.bar >= len(self.data):
            # 终止条件①：数据走完了，正常结束
            done = True
        elif reward == 1:
            # 终止条件②：刚猜对 → 继续跑（即使准确率低，只要还能猜对就不提前终止）
            done = False
        elif (self.accuracy < self.min_accuracy and
              self.bar > self.n_features + 15):
            # 终止条件③：准确率跌破阈值，且已经跑了至少 n_features + 15 步
            # 多了 n_features 是因为从 bar=n_features 才开始真正预测，
            # 所以"至少跑了 15 步"其实是指 (bar - n_features) > 15
            done = True
        else:
            done = False

        # 构造下一状态：与 _get_state() 逻辑相同，取最近 n_features 根K线
        next_state = self.data_[self.feature].iloc[
            self.bar - self.n_features:self.bar].values

        # 返回 Gym 标准五元组
        return next_state, reward, done, False, {}
#
# Finance Environment with Historical Data
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#
# 功能概述：
#   实现一个基于真实历史金融数据的强化学习环境。
#   核心任务是"涨跌方向预测"——智能体需要根据历史价格特征，
#   预测下一个时间周期的收益率是正（上涨）还是负（下跌）。
#
#   环境使用的数据源是 TPQ（The Python Quants）官网提供的 CSV 文件，
#   其中包含 GLD（SPDR 黄金 ETF）等资产的历史行情。
#

import random
import numpy as np
import pandas as pd


class ActionSpace:
    """
    动作空间类——定义智能体可以执行的所有动作。

    本环境是"涨跌方向预测"任务，因此只有 2 个动作：
      action = 0  → 预测下一根K线下跌
      action = 1  → 预测下一根K线上涨

    注意：这不是交易中的"多空持仓"动作，
    而是纯粹的二分类预测动作。
    """

    # 动作总数：0 和 1 两种选择
    n = 2

    def sample(self):
        """
        从动作空间中随机采样一个动作。
        用于训练初期的探索（epsilon-greedy 策略）。

        Returns:
            int: 0 或 1，代表随机选择的预测方向
        """
        return random.randint(0, 1)


class Finance:
    """
    金融强化学习环境——基于真实历史数据的涨跌预测。

    数据流程：
      1. 从远程 CSV 下载原始行情数据（self._get_data）
      2. 提取指定标的的价格序列，计算对数收益率和涨跌标记（self._prepare_data）
      3. 标准化特征数据，供智能体观察（self.data_）

    交互流程（符合 Gym 风格）：
      reset()  → 初始化环境，返回初始状态
      step(action) → 执行动作，返回 (下一状态, 奖励, 是否结束, 截断标志, 额外信息)

    Attributes:
        symbol (str): 要交易/预测的标的名称，如 'GLD'
        feature (str): 用作观察特征的列名，如 'r' 表示用收益率作为特征
        n_features (int): 每次观察的历史窗口长度（多少根K线）
        action_space (ActionSpace): 动作空间对象
        min_accuracy (float): 提前终止阈值——准确率低于此值时 episode 提前结束
        raw (pd.DataFrame): 从 CSV 加载的原始数据
        data (pd.DataFrame): 处理后的数据，包含价格、收益率、涨跌标记
        data_ (pd.DataFrame): 标准化后的数据，作为智能体的观察输入
        bar (int): 当前进度指针，指向正在预测的那根K线
        treward (int): 本 episode 累计获得的正确预测次数
        accuracy (float): 本 episode 当前的预测准确率
    """

    # 类属性：数据源 URL
    # 远程 CSV 文件，包含 GLD 等标的的历史行情数据
    url = './03_samples.csv'
    # url = 'https://certificate.tpq.io/rl4finance.csv'

    def __init__(self, symbol, feature, min_accuracy=0.485, n_features=4):
        """
        初始化 Finance 环境。

        Args:
            symbol (str): 要分析的标的符号，如 'GLD'（黄金ETF）
            feature (str): 作为观察特征的列名，如 'r'（对数收益率）
            min_accuracy (float, optional): 提前终止的最低准确率阈值。
                如果智能体的预测准确率持续低于此值，episode 会提前结束，
                避免浪费时间在表现太差的轨迹上。默认 0.485。
            n_features (int, optional): 观察窗口长度——智能体每次能看到
                多少根历史K线的数据来做预测。默认 4。
        """
        self.symbol = symbol
        self.feature = feature
        self.n_features = n_features
        self.action_space = ActionSpace()
        self.min_accuracy = min_accuracy

        # 两步数据准备：先下载原始数据，再做预处理
        self._get_data()
        self._prepare_data()

    def _get_data(self):
        """
        从远程 URL 下载原始行情数据。

        数据源是 TPQ 官网提供的 CSV 文件，格式为：
          - 索引：日期时间（parse_dates=True 自动解析）
          - 列：不同标的的收盘价（如 'GLD', 'AAPL' 等）

        下载后存储到 self.raw，供 _prepare_data 进一步处理。
        """
        self.raw = pd.read_csv(self.url,
                index_col=0, parse_dates=True)

    def _prepare_data(self):
        """
        对原始数据进行预处理，构建环境所需的全部特征。

        处理步骤：
          1. 从原始数据中提取指定标的（self.symbol）的收盘价，去除缺失值
          2. 计算对数收益率 r = ln(S_t / S_{t-1})，比简单收益率更适合金融建模
          3. 生成涨跌方向标记 d：r > 0 → 1（上涨），r ≤ 0 → 0（下跌）
             这就是智能体要预测的"标签"
          4. 再次删除计算收益率后产生的 NaN（第一行没有前一天数据）
          5. 对所有列做 Z-score 标准化：(x - mean) / std
             标准化后的数据存到 self.data_，作为智能体的观察输入
             原始未标准化的数据保留在 self.data，用于计算真实标签和 P&L

        最终 self.data 包含：
          [symbol列（收盘价）, 'r'（对数收益率）, 'd'（涨跌标记）]
        最终 self.data_ 包含：
          同 self.data 的列，但全部做了标准化
        """
        # 提取目标标的的收盘价序列
        self.data = pd.DataFrame(self.raw[self.symbol]).dropna()

        # 计算对数收益率：r_t = ln(S_t / S_{t-1})
        # 对数收益率具有可加性（多期收益率 = 单期收益率之和），且更接近正态分布
        self.data['r'] = np.log(self.data / self.data.shift(1))

        # 生成二分类标签：1 表示上涨（收益率>0），0 表示下跌（收益率≤0）
        # 这就是智能体需要预测的"ground truth"
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)

        # 删除计算收益率后第一行产生的 NaN
        self.data.dropna(inplace=True)

        # Z-score 标准化：将所有特征缩放到均值为0、标准差为1的分布
        # 标准化是神经网络训练的标准预处理，能加速收敛、避免梯度爆炸
        self.data_ = (self.data - self.data.mean()) / self.data.std()

    def reset(self):
        """
        重置环境到初始状态，开始一个新的 episode。

        具体操作：
          1. 将 bar 指针重置为 n_features（前面的K线用来构建初始观察窗口）
          2. 累计奖励归零
          3. 返回初始状态——从标准化数据中截取前 n_features 行的特征序列

        Returns:
            tuple: (state, info)
              - state (np.ndarray): 形状为 (n_features,) 的一维数组，
                包含前 n_features 根K线的特征值，作为智能体的初始观察
              - info (dict): 辅助信息字典（Gym 标准接口），本环境返回空字典
        """
        # bar 从 n_features 开始，因为前 n_features 根K线用来构造观察窗口，
        # 智能体需要"看完"这些历史数据后，才能对第 n_features+1 根K线做预测
        self.bar = self.n_features
        self.treward = 0

        # 构造初始观察：取 self.feature 列中 [bar - n_features, bar) 的切片
        # 即最后 n_features 根历史K线的特征值
        state = self.data_[self.feature].iloc[
            self.bar - self.n_features:self.bar].values

        return state, {}

    def step(self, action):
        """
        执行一个动作，推进环境一个时间步。

        这是强化学习的核心交互函数——智能体根据当前状态选择一个动作，
        环境返回执行结果（下一状态、奖励、是否结束）。

        本环境的 step 逻辑：
          1. 检查智能体的预测（action）是否与真实涨跌方向（self.data['d']）一致
             - 正确：reward = 1
             - 错误：reward = 0
          2. 更新累计奖励和当前准确率
          3. 判断 episode 是否结束（三种终止条件，见下方）
          4. 推进 bar 指针，返回下一状态

        episode 终止条件（满足任一即终止）：
          ① bar 走到数据末尾 → 正常结束，整条历史数据用完了
          ② 准确率低于 min_accuracy 且已经跑了至少 15 步 → 提前止损，
             避免在表现太差的轨迹上浪费计算资源
          ③ 刚刚预测正确（reward == 1）→ 继续，不触发提前终止
             （注意：这是一个有趣的设计——只要还能猜对，就不给提前判死刑）

        Args:
            action (int): 智能体选择的动作，0 或 1
                - 0 → 预测当前 bar 下跌
                - 1 → 预测当前 bar 上涨

        Returns:
            tuple: (next_state, reward, done, truncated, info)
              - next_state (np.ndarray): 下一时刻的观察，形状同 reset 的返回
              - reward (int): 本步奖励，1 表示预测正确，0 表示预测错误
              - done (bool): episode 是否结束
              - truncated (bool): 是否因超时/步数限制截断（Gym 标准接口），本环境固定 False
              - info (dict): 辅助信息，本环境返回空字典
        """
        # 判断预测是否正确：
        # self.data['d'].iloc[self.bar] 是当前 bar 的真实涨跌标记（ground truth）
        # action 是智能体的预测
        if action == self.data['d'].iloc[self.bar]:
            correct = True
        else:
            correct = False

        # 本步奖励：二值奖励，正确得 1 分，错误得 0 分
        # 注意：这里没有用 P&L 作为奖励，纯粹是"准确率"奖励
        # 这样设计的好处是避免 agent 为了赚大钱而冒大险
        reward = 1 if correct else 0

        # 更新累计奖励和进度指针
        self.treward += reward
        self.bar += 1

        # 计算当前准确率 = 累计正确次数 / 已预测的步数
        # 分母是 (bar - n_features)，因为前 n_features 步是用来构建初始窗口的，
        # 真正开始预测是从第 n_features 步之后
        self.accuracy = self.treward / (self.bar - self.n_features)

        # ===== 判断 episode 是否结束 =====
        if self.bar >= len(self.data):
            # 终止条件①：数据走完了，正常结束
            done = True
        elif reward == 1:
            # 终止条件③：刚猜对，继续跑（即使准确率低，只要还能猜对就不提前终止）
            done = False
        elif (self.accuracy < self.min_accuracy) and (self.bar > 15):
            # 终止条件②：准确率跌破阈值，且已经跑了至少 15 步（避免刚开局就误判）
            # → 提前终止，这条轨迹表现太差，不值得继续
            done = True
        else:
            done = False

        # 构造下一状态：和 reset 里逻辑一样，截取最近 n_features 根K线的特征
        next_state = self.data_[self.feature].iloc[
            self.bar - self.n_features:self.bar].values

        # 返回 Gym 标准的五元组
        # truncated=False 表示本环境没有步数上限截断（用 done 管理终止）
        return next_state, reward, done, False, {}
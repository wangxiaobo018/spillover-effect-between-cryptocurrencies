
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import norm
from scipy.special import logsumexp
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mstats, norm
from scipy.special import gamma
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, KFold
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import t, kendalltau, spearmanr
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE

from copulae import TCopula
# 读取数据
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
}


# 定义计算收益率的函数
def calculate_returns(prices):
    # 计算日收益率：(P_t / P_{t-1} - 1) * 100
    returns = (prices / prices.shift(1) - 1) * 100
    returns.iloc[0] = 0  # 第一个收益率设为0
    returns[prices.shift(1) == 0] = np.nan  # 处理除零情况
    return returns


# 定义计算RV的函数
def calculate_rv(df, coin_name):
    # 复制数据并重命名列
    data_ret = df[['time', 'close']].copy()
    data_ret.columns = ['DT', 'PRICE']
    data_ret = data_ret.dropna()  # 删除缺失值

    # 计算收益率
    data_ret['Ret'] = calculate_returns(data_ret['PRICE'])

    # 将DT转换为日期
    data_ret['DT'] = pd.to_datetime(data_ret['DT']).dt.date

    # 计算日度RV：收益率平方的日度总和
    RV = (data_ret
          .groupby('DT')['Ret']
          .apply(lambda x: np.sum(x ** 2))
          .reset_index())

    # 重命名RV列为币种名称
    RV.columns = ['DT', f'RV_{coin_name}']
    return RV


# 计算每种加密货币的RV
rv_dfs = []
for coin, file_path in data_files.items():
    df = pd.read_csv(file_path)
    rv_df = calculate_rv(df, coin)
    rv_dfs.append(rv_df)

# 合并所有RV数据框，按DT对齐
rv_merged = rv_dfs[0]  # 以第一个RV数据框（BTC）为基础
for rv_df in rv_dfs[1:]:
    rv_merged = rv_merged.merge(rv_df, on='DT', how='outer')

# 将DT转换为datetime格式（可选）
rv_merged['DT'] = pd.to_datetime(rv_merged['DT'])

# 按日期排序
all_RV = rv_merged.sort_values('DT').reset_index(drop=True)

all_RV= all_RV.dropna()  # 删除包含NaN的行

# 数据准备
all_RV.columns = ["DT","BTC", "DASH","ETH","LTC","XLM","XRP"]

features = all_RV.drop(columns=['DT'])
data = features.to_numpy()

columns = ["BTC", "DASH","ETH","LTC","XLM","XRP"]
data_df = pd.DataFrame(data, columns=columns)

# 参数设置
# 参数设置
# 参数设置
n = len(columns)  # 变量数
p = 1  # 滞后阶数
H = 12  # 预测步长
kappa1 = 0.99  # 遗忘因子 1
kappa2 = 0.96  # 遗忘因子 2
T = data.shape[0]  # 时间长度

# 初始化 TVP-VAR 参数
var_model = VAR(data_df[:3]).fit(maxlags=p)
A_0 = np.hstack([coef for coef in var_model.coefs])  # 初始系数矩阵
Sigma_0 = np.cov(data[:3], rowvar=False)  # 初始协方差矩阵
Sigma_A_0 = np.eye(n * p) * 0.1  # 初始系数协方差


# TVP-VAR 模型（Kalman 滤波）
def tvp_var(y, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0):
    T, n = y.shape
    A_t = np.zeros((n, n * p, T))  # 时间变化系数
    Sigma_t = np.zeros((n, n, T))  # 时间变化协方差
    Sigma_A_t = np.zeros((n * p, n * p, T))  # 系数协方差

    A_t[:, :, 0] = A_0
    Sigma_t[:, :, 0] = Sigma_0
    Sigma_A_t[:, :, 0] = Sigma_A_0

    for t in range(1, T):
        z_t = y[t - 1:t - p:-1].flatten()
        if len(z_t) < n * p:
            z_t = np.pad(z_t, (0, n * p - len(z_t)), 'constant')

        A_t_pred = A_t[:, :, t - 1]
        Sigma_A_t_pred = Sigma_A_t[:, :, t - 1] + (1 / kappa1 - 1) * Sigma_A_t[:, :, t - 1]
        epsilon_t = y[t - 1] - A_t_pred @ z_t
        Sigma_t_pred = kappa2 * Sigma_t[:, :, t - 1] + (1 - kappa2) * np.outer(epsilon_t, epsilon_t)

        K_t = Sigma_A_t_pred @ z_t @ np.linalg.pinv(z_t @ Sigma_A_t_pred @ z_t.T + Sigma_t_pred)
        A_t[:, :, t] = A_t_pred + K_t @ (y[t] - A_t_pred @ z_t)
        Sigma_A_t[:, :, t] = (np.eye(n * p) - K_t @ z_t) @ Sigma_A_t_pred
        epsilon_t_updated = y[t] - A_t[:, :, t] @ z_t
        Sigma_t[:, :, t] = kappa2 * Sigma_t[:, :, t - 1] + (1 - kappa2) * np.outer(epsilon_t_updated, epsilon_t_updated)

    return A_t, Sigma_t, Sigma_A_t


# 运行 TVP-VAR
A_t, Sigma_t, Sigma_A_t = tvp_var(data, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0)


# 计算广义预测误差方差分解（GFEVD）
def gfevd(A_t, Sigma_t, H, n):
    T = A_t.shape[2]
    gfevd_array = np.zeros((n, n, T))

    for t in range(T):
        B_jt = np.zeros((n, n, H))
        M_t = np.zeros((n * p, n * p))
        M_t[:n, :n * p] = A_t[:, :, t]
        M_t[n:, :-n] = np.eye(n * (p - 1))
        J = np.vstack((np.eye(n), np.zeros((n * (p - 1), n))))

        for j in range(H):
            B_jt[:, :, j] = J.T @ np.linalg.matrix_power(M_t, j) @ J

        Psi_t = np.zeros((n, n, H))
        for h in range(H):
            Psi_t[:, :, h] = B_jt[:, :, h] @ Sigma_t[:, :, t] @ np.diag(1 / np.sqrt(np.diag(Sigma_t[:, :, t])))

        gfevd_t = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gfevd_t[i, j] = np.sum(Psi_t[i, j, :] ** 2) / np.sum(Psi_t[i, :, :] ** 2)
        gfevd_t = gfevd_t / gfevd_t.sum(axis=1, keepdims=True)
        gfevd_array[:, :, t] = gfevd_t

    return gfevd_array


# 计算 GFEVD
gfevd_results = gfevd(A_t, Sigma_t, H, n)

def npdc_to_btc(gfevd_array, columns):
    T = gfevd_array.shape[2]
    n = len(columns)
    npdc_btc = np.zeros((T, n))

    for t in range(T):
        for j in range(n):
            npdc_btc[t, j] = (gfevd_array[0, j, t] - gfevd_array[j, 0, t]) * 100  # 假设 BTC 是第 0 列

    npdc_df = pd.DataFrame(npdc_btc, columns=[f"NPDC_{col}_BTC" for col in columns])
    other_cols = [f"NPDC_{col}_BTC" for col in columns if col != 'BTC']
    npdc_df['Total_NPDC_others_BTC'] = npdc_df[other_cols].sum(axis=1)
    return npdc_df

# 计算总溢出效应
def total_spillover(gfevd_array):
    T = gfevd_array.shape[2]
    n = gfevd_array.shape[0]
    total_spillover_values = np.zeros(T)

    for t in range(T):
        gfevd_t = gfevd_array[:, :, t]
        numerator = np.sum(gfevd_t) - np.trace(gfevd_t)
        denominator = np.sum(gfevd_t)
        total_spillover_values[t] = (numerator / denominator) * 100

    return pd.Series(total_spillover_values, name="Total_Spillover")
npdc_df = npdc_to_btc(gfevd_results, columns)
print(npdc_df)


def npdc_all_pairs(gfevd_array, columns):
    T = gfevd_array.shape[2]
    n = len(columns)
    npdc_dict = {}

    # 计算所有变量对之间的 NPDC
    for i in range(n):
        for j in range(n):
            if i != j:
                npdc_values = np.zeros(T)
                for t in range(T):
                    npdc_values[t] = (gfevd_array[i, j, t] - gfevd_array[j, i, t]) * 100
                npdc_dict[f"NPDC_{columns[i]}_to_{columns[j]}"] = npdc_values

    # 转换为 DataFrame
    npdc_df = pd.DataFrame(npdc_dict)
    return npdc_df

# 计算所有变量对的 NPDC
columns = ["BTC", "DASH", "ETH", "LTC", "XLM", "XRP"]
npdc_df_all = npdc_all_pairs(gfevd_results, columns)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# 选择最后一个时间点
t = gfevd_results.shape[2] - 1

# 创建有向图
G = nx.DiGraph()

# 添加节点（你的变量）
nodes = columns  # ["BTC", "DASH", "ETH", "LTC", "XLM", "XRP"]
for node in nodes:
    G.add_node(node)

# 计算每个节点的净溢出效应（总 NPDC）
net_spillover = np.zeros(len(nodes))
for i, node_i in enumerate(nodes):
    total_npdcs = 0
    for j, node_j in enumerate(nodes):
        if i != j:
            npdc_value = (gfevd_results[i, j, t] - gfevd_results[j, i, t]) * 100
            total_npdcs += npdc_value
    net_spillover[i] = total_npdcs

# 确定节点颜色（净发射器为蓝色，净接收器为黄色）
# 确定节点颜色（净发射器为天蓝色，净接收器为淡黄色）
node_colors = {}
for i, node in enumerate(nodes):
    if net_spillover[i] > 0:
        node_colors[node] = "#A9D0F5"  # 天蓝色
    else:
        node_colors[node] = "#FFE39D"  # 淡黄色

# 动态调整节点大小（基于净溢出效应的绝对值）
min_size = 500  # 最小节点大小
max_size = 2000  # 最大节点大小
node_sizes = min_size + (max_size - min_size) * (np.abs(net_spillover) / np.max(np.abs(net_spillover)))

# 添加边（基于 NPDC 值，并计算厚度）
edges = []
widths = []
for i, node_i in enumerate(nodes):
    for j, node_j in enumerate(nodes):
        if i != j:
            npdc_value = (gfevd_results[i, j, t] - gfevd_results[j, i, t]) * 100
            if npdc_value > 0:  # 只绘制正的净溢出
                edges.append((node_i, node_j))
                widths.append(np.abs(npdc_value) / 10)  # 调整粗细比例

# 绘制网络图
plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G)  # 使用圆形布局，避免箭头重叠

# 绘制节点（大小基于净溢出效应）
nx.draw_networkx_nodes(G, pos, node_color=[node_colors[node] for node in G.nodes()], node_size=node_sizes)

# 绘制边（带箭头，厚度基于 NPDC）
nx.draw_networkx_edges(
    G, pos,
    edgelist=edges,
    width=widths,
    arrows=True,
    arrowstyle="->",
    arrowsize=20,
    connectionstyle="arc3,rad=0.1"
)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")


plt.axis("off")
# 保存图形
plt.savefig("network_graph.png", dpi=300, bbox_inches="tight")
plt.show()


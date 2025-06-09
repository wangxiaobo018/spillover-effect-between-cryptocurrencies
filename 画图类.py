# import numpy as np
# import pandas as pd
# from scipy.optimize import differential_evolution, minimize
# from concurrent.futures import ThreadPoolExecutor
# import warnings
# import seaborn as sns
# from statsmodels.tsa.vector_ar.var_model import VAR
# from scipy.stats import norm
# from scipy.special import logsumexp
# warnings.filterwarnings('ignore')
# import statsmodels.api as sm
# from statsmodels.regression.linear_model import OLS
# from sklearn.linear_model import LinearRegression, LassoCV, Lasso
# from scipy.ndimage import uniform_filter1d
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy.stats import mstats, norm
# from scipy.special import gamma
# from tqdm import tqdm
# from sklearn.model_selection import TimeSeriesSplit, KFold
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from sklearn.preprocessing import StandardScaler
# import os
# from scipy.stats import norm
# from scipy.special import logsumexp
# import json
# from numpy.linalg import inv
# import os
# from statsmodels.tsa.vector_ar.var_model import VAR
# # Read the data
# # 读取数据
# data_files = {
#     'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
#     'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
#     'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
#     'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
#     'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
#     'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
# }
#
#
# # 定义计算收益率的函数
# def calculate_returns(prices):
#     # 计算日收益率：(P_t / P_{t-1} - 1) * 100
#     returns = (prices / prices.shift(1) - 1) * 100
#     returns.iloc[0] = 0  # 第一个收益率设为0
#     returns[prices.shift(1) == 0] = np.nan  # 处理除零情况
#     return returns
#
#
# # 定义计算RV的函数
# def calculate_rv(df, coin_name):
#     # 复制数据并重命名列
#     data_ret = df[['time', 'close']].copy()
#     data_ret.columns = ['DT', 'PRICE']
#     data_ret = data_ret.dropna()  # 删除缺失值
#
#     # 计算收益率
#     data_ret['Ret'] = calculate_returns(data_ret['PRICE'])
#
#     # 将DT转换为日期
#     data_ret['DT'] = pd.to_datetime(data_ret['DT']).dt.date
#
#     # 计算日度RV：收益率平方的日度总和
#     RV = (data_ret
#           .groupby('DT')['Ret']
#           .apply(lambda x: np.sum(x ** 2))
#           .reset_index())
#
#     # 重命名RV列为币种名称
#     RV.columns = ['DT', f'RV_{coin_name}']
#     return RV
#
#
# # 计算每种加密货币的RV
# rv_dfs = []
# for coin, file_path in data_files.items():
#     df = pd.read_csv(file_path)
#     rv_df = calculate_rv(df, coin)
#     rv_dfs.append(rv_df)
#
# # 合并所有RV数据框，按DT对齐
# rv_merged = rv_dfs[0]  # 以第一个RV数据框（BTC）为基础
# for rv_df in rv_dfs[1:]:
#     rv_merged = rv_merged.merge(rv_df, on='DT', how='outer')
#
# # 将DT转换为datetime格式（可选）
# rv_merged['DT'] = pd.to_datetime(rv_merged['DT'])
#
# # 按日期排序
# all_RV = rv_merged.sort_values('DT').reset_index(drop=True)
#
# all_RV= all_RV.dropna()  # 删除包含NaN的行
#
# # 数据准备
# all_RV.columns = ["DT","BTC", "DASH","ETH","LTC","XLM","XRP"]
#
# features = all_RV.drop(columns=['DT'])
# data = features.to_numpy()
#
# columns = ["BTC", "DASH","ETH","LTC","XLM","XRP"]
# data_df = pd.DataFrame(data, columns=columns)
# print(all_RV)
# # 参数设置
# # 参数设置
# n = len(columns)  # 变量数
# p = 1  # 滞后阶数
# H = 12  # 预测步长
# kappa1 = 0.99  # 遗忘因子 1
# kappa2 = 0.96  # 遗忘因子 2
# T = data.shape[0]  # 时间长度
#
# # 初始化 TVP-VAR 参数
# var_model = VAR(data_df[:3]).fit(maxlags=p)
# A_0 = np.hstack([coef for coef in var_model.coefs])  # 初始系数矩阵
# Sigma_0 = np.cov(data[:3], rowvar=False)  # 初始协方差矩阵
# Sigma_A_0 = np.eye(n * p) * 0.1  # 初始系数协方差
#
#
# # TVP-VAR 模型（Kalman 滤波）
# def tvp_var(y, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0):
#     T, n = y.shape
#     A_t = np.zeros((n, n * p, T))  # 时间变化系数
#     Sigma_t = np.zeros((n, n, T))  # 时间变化协方差
#     Sigma_A_t = np.zeros((n * p, n * p, T))  # 系数协方差
#
#     A_t[:, :, 0] = A_0
#     Sigma_t[:, :, 0] = Sigma_0
#     Sigma_A_t[:, :, 0] = Sigma_A_0
#
#     for t in range(1, T):
#         z_t = y[t - 1:t - p:-1].flatten()
#         if len(z_t) < n * p:
#             z_t = np.pad(z_t, (0, n * p - len(z_t)), 'constant')
#
#         A_t_pred = A_t[:, :, t - 1]
#         Sigma_A_t_pred = Sigma_A_t[:, :, t - 1] + (1 / kappa1 - 1) * Sigma_A_t[:, :, t - 1]
#         epsilon_t = y[t - 1] - A_t_pred @ z_t
#         Sigma_t_pred = kappa2 * Sigma_t[:, :, t - 1] + (1 - kappa2) * np.outer(epsilon_t, epsilon_t)
#
#         K_t = Sigma_A_t_pred @ z_t @ np.linalg.pinv(z_t @ Sigma_A_t_pred @ z_t.T + Sigma_t_pred)
#         A_t[:, :, t] = A_t_pred + K_t @ (y[t] - A_t_pred @ z_t)
#         Sigma_A_t[:, :, t] = (np.eye(n * p) - K_t @ z_t) @ Sigma_A_t_pred
#         epsilon_t_updated = y[t] - A_t[:, :, t] @ z_t
#         Sigma_t[:, :, t] = kappa2 * Sigma_t[:, :, t - 1] + (1 - kappa2) * np.outer(epsilon_t_updated, epsilon_t_updated)
#
#     return A_t, Sigma_t, Sigma_A_t
#
#
# # 运行 TVP-VAR
# A_t, Sigma_t, Sigma_A_t = tvp_var(data, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0)
#
#
# # 计算广义预测误差方差分解（GFEVD）
# def gfevd(A_t, Sigma_t, H, n):
#     T = A_t.shape[2]
#     gfevd_array = np.zeros((n, n, T))
#
#     for t in range(T):
#         B_jt = np.zeros((n, n, H))
#         M_t = np.zeros((n * p, n * p))
#         M_t[:n, :n * p] = A_t[:, :, t]
#         M_t[n:, :-n] = np.eye(n * (p - 1))
#         J = np.vstack((np.eye(n), np.zeros((n * (p - 1), n))))
#
#         for j in range(H):
#             B_jt[:, :, j] = J.T @ np.linalg.matrix_power(M_t, j) @ J
#
#         Psi_t = np.zeros((n, n, H))
#         for h in range(H):
#             Psi_t[:, :, h] = B_jt[:, :, h] @ Sigma_t[:, :, t] @ np.diag(1 / np.sqrt(np.diag(Sigma_t[:, :, t])))
#
#         gfevd_t = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 gfevd_t[i, j] = np.sum(Psi_t[i, j, :] ** 2) / np.sum(Psi_t[i, :, :] ** 2)
#         gfevd_t = gfevd_t / gfevd_t.sum(axis=1, keepdims=True)
#         gfevd_array[:, :, t] = gfevd_t
#
#     return gfevd_array
#
#
# # 计算 GFEVD
# gfevd_results = gfevd(A_t, Sigma_t, H, n)
#
# def npdc_to_btc(gfevd_array, columns):
#     T = gfevd_array.shape[2]
#     n = len(columns)
#     npdc_btc = np.zeros((T, n))
#
#     for t in range(T):
#         for j in range(n):
#             npdc_btc[t, j] = (gfevd_array[0, j, t] - gfevd_array[j, 0, t]) * 100  # 假设 BTC 是第 0 列
#
#     npdc_df = pd.DataFrame(npdc_btc, columns=[f"NPDC_{col}_BTC" for col in columns])
#     other_cols = [f"NPDC_{col}_BTC" for col in columns if col != 'BTC']
#     npdc_df['Total_NPDC_others_BTC'] = npdc_df[other_cols].sum(axis=1)
#     return npdc_df
#
# # 计算总溢出效应
# def total_spillover(gfevd_array):
#     T = gfevd_array.shape[2]
#     n = gfevd_array.shape[0]
#     total_spillover_values = np.zeros(T)
#
#     for t in range(T):
#         gfevd_t = gfevd_array[:, :, t]
#         numerator = np.sum(gfevd_t) - np.trace(gfevd_t)
#         denominator = np.sum(gfevd_t)
#         total_spillover_values[t] = (numerator / denominator) * 100
#
#     return pd.Series(total_spillover_values, name="Total_Spillover")
# # 计算各市场对SSE的净溢出效应
# npdc_df = npdc_to_btc(gfevd_results, columns)
# print(npdc_df)
# npdc = npdc_df.iloc[:, 1:-1]
# # 选择需要分析的列（排除 SSE 自身和总和列）
#
# # 计算哪个市场对 BTC 溢出最多
# npdc_columns = ['NPDC_DASH_BTC', 'NPDC_ETH_BTC', 'NPDC_LTC_BTC', 'NPDC_XLM_BTC', 'NPDC_XRP_BTC']
# max_spillover_to_btc = npdc_df[npdc_columns].apply(lambda x: x.idxmax() if x.max() > 0 else None, axis=1)
# max_spillover_to_btc_counts = max_spillover_to_btc.value_counts()
#
# # 输出结果
# print("=== 哪个市场对 BTC 溢出最多 ===")
# print("各市场对 BTC 溢出最多的频率：")
# print(max_spillover_to_btc_counts)
# print("\n平均 NPDC 值（正值表示对 BTC 的净溢出）：")
# print(npdc_df[npdc_columns].mean())
#
#
#
#
# # # 可视化NPDC随时间变化
# # import matplotlib.pyplot as plt
# #
# # plt.figure(figsize=(12, 6))
# # for col in npdc_columns:
# #     plt.plot(npdc_df.index, npdc_df[col], label=col)
# # plt.axhline(0, color='black', linestyle='--')
# # plt.title('Net Pairwise Directional Connectedness to/from SSE')
# # plt.xlabel('Time')
# # plt.ylabel('NPDC (%)')
# # plt.legend()
# # plt.show()
#
# common_columns = all_RV.columns.intersection(npdc.columns)
#
# # 遍历共同列并更新值
# data1_modified = all_RV.copy()
# print(data1_modified)
# #
# # # 遍历共同列并更新值
# # for col in common_columns:
# #     if col != 'Date':  # 跳过 Date 列
# #         # 使用 vectorized operation 更新值
# #         data1_modified[col] = np.where(npdc[col] < 0, all_RV[col], 0)
# #
# data1_modified = data1_modified.rename(columns={"BTC": "RV"})
# # print(data1_modified)
#
#
# # 数据准备 (从您提供的代码)
# model1 = pd.DataFrame({
#     'DT':all_RV['DT'],
#     'RV': data1_modified['RV'],
#     'rv_lag1': data1_modified['RV'].shift(1),
#     'rv_lag5': data1_modified['RV'].rolling(window=5).mean().shift(1),
#     'rv_lag22': data1_modified['RV'].rolling(window=22).mean().shift(1),
#     'BTC_lag1': data1_modified['LTC'].shift(1)
# }).dropna()

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.optimize import differential_evolution, minimize
from concurrent.futures import ThreadPoolExecutor
import warnings
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import norm
from scipy.special import logsumexp
import statsmodels.tools.numdiff as nd
from statsmodels.tools.eval_measures import aic, bic, hqic
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
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv
from scipy.stats import t, kendalltau, spearmanr
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE

# 定义文件路径字典
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/dash.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/eth.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ltc.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/xlm.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/xrp.csv"
}

# 定义 get_RV_BV 函数
def get_RV_BV(data, alpha=0.05, times=True):
    idx = 100 if times else 1
    df = data.copy()

    df['datetime'] = pd.to_datetime(df['time'])
    df['day'] = df['datetime'].dt.date

    results = []
    for day, group in df.groupby('day'):
        group = group.sort_values('datetime')

        group['Ret'] = (np.log(group['close']) - np.log(group['close'].shift(1))) * idx

        group = group.dropna(subset=['Ret'])
        n = len(group)

        if n < 5:
            continue

        # 计算 RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算 BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())

        # 计算 TQ
        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))

        term1 = abs_ret.iloc[4:].values
        term2 = abs_ret.iloc[2:-2].values
        term3 = abs_ret.iloc[:-4].values

        min_len = min(len(term1), len(term2), len(term3))
        if min_len > 0:
            TQ = TQ_coef * np.sum((term1[:min_len] ** (4 / 3)) *
                                  (term2[:min_len] ** (4 / 3)) *
                                  (term3[:min_len] ** (4 / 3)))
        else:
            continue

        # Z_test
        Z_test = ((RV - BV) / RV) / np.sqrt(((np.pi / 2) ** 2 + np.pi - 5) *
                                            (1 / n) * max(1, TQ / (BV ** 2)))

        # 计算 JV 和 C_t
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({
            'day': day,
            'C_t': C_t,
            'JV': JV
        })

    result_df = pd.DataFrame(results)
    return result_df

# 创建字典来存储 C_t 和 JV 数据
c_t_dict = {}
jv_dict = {}

# 遍历每个数据文件，计算 HAR-CJ
for code, file_path in data_files.items():
    # 读取数据
    df = pd.read_csv(file_path)

    # 筛选数据
    data_filtered = df[df['code'] == code].copy()

    # 计算 HAR-CJ
    har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)

    # 将 day 转换为字符串，以便合并时使用
    har_cj['day'] = har_cj['day'].astype(str)

    # 存储 C_t 和 JV 数据
    c_t_dict[code] = har_cj[['day', 'C_t']].set_index('day')['C_t']
    jv_dict[code] = har_cj[['day', 'JV']].set_index('day')['JV']

# 转换为 DataFrame
ct = pd.DataFrame(c_t_dict)
jv = pd.DataFrame(jv_dict)


all_CT = ct.reset_index()  # Resets the 'day' index to a column
all_CT = all_CT.rename(columns={'day': 'DT'})  # Renames the 'day' column to 'DT'
print(all_CT)
all_JV = jv.reset_index()  # Resets the 'day' index to a column
all_JV = all_JV.rename(columns={'day': 'DT'})  # Renames the 'day' column to 'DT'
print(all_JV)



# Read the data
df_data = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")

# Get group summary
group_summary = df_data.groupby('code').size().reset_index(name='NumObservations')

# Create data_ret DataFrame with renamed columns first
data_ret = df_data[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()

# Calculate returns for each group using the new formula
def calculate_returns(prices):
    # Compute daily returns using the given formula
    returns = (prices / prices.shift(1) - 1)*100
    returns.iloc[0] = 0  # First return is 0
    returns[prices.shift(1) == 0] = np.nan  # Handle division by zero
    return returns

# Calculate returns by group
data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)

# Get group summary for data_ret
group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')

# Filter for "000001.XSHG" and remove unnecessary columns
data_filtered = data_ret[data_ret['id'] == "BTC"].copy()
data_filtered = data_filtered.drop('id', axis=1)
from datetime import date
# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
# Filter using date objects
data_filtered = data_filtered[
    (data_filtered['DT'] >= date(2019, 3, 28)) &
    (data_filtered['DT'] <= date(2025, 3, 30))
]
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']
# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])


JV_lag1 = all_JV['BTC'].shift(1)
C_t_lag1 = all_CT['BTC'].shift(1)
JV_lag5 = all_JV['BTC'].rolling(window=5).mean().shift(1)
C_t_lag5 = all_CT['BTC'].rolling(window=5).mean().shift(1)
JV_lag22 = all_JV['BTC'].rolling(window=22).mean().shift(1)
C_t_lag22 = all_CT['BTC'].rolling(window=22).mean().shift(1)

model1 = pd.DataFrame({
    'DT': RV['DT'],
    'RV': RV['RV'],
    'JV_lag1': JV_lag1,
    'C_t_lag1': C_t_lag1,
    'JV_lag5': JV_lag5,
    'C_t_lag5': C_t_lag5,
    'JV_lag22': JV_lag22,
    'C_t_lag22': C_t_lag22,
    'BTC_lag1': all_CT['LTC'].shift(1)  # Assuming BTC_lag1 is the lagged RV
}).dropna()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


class TimeVaryingMarkovSwitching:
    """
    时变马尔可夫转换模型
    转换概率依赖于过渡变量
    """

    def __init__(self, n_regimes=2):
        self.n_regimes = n_regimes
        self.params = None
        self.smoothed_probs = None
        self.filtered_probs = None
        self.transition_params = None

    def _time_varying_transition_matrix(self, z_t, gamma):
        """
        计算时变转换概率矩阵
        使用logistic函数
        """
        P = np.zeros((2, 2))

        # 从状态0到状态1的概率
        P[0, 1] = expit(gamma[0] + gamma[1] * z_t)
        P[0, 0] = 1 - P[0, 1]

        # 从状态1到状态0的概率
        P[1, 0] = expit(gamma[2] + gamma[3] * z_t)
        P[1, 1] = 1 - P[1, 0]

        return P

    def _log_likelihood(self, params, y, z):
        """计算对数似然函数"""
        T = len(y)

        # 解包参数
        mu = params[0:2]
        sigma = np.abs(params[2:4])
        gamma = params[4:8]

        # 初始概率
        pi = np.array([0.5, 0.5])

        # 前向算法
        log_likelihood = 0
        alpha = np.zeros((T, 2))

        # t=0
        P_0 = self._time_varying_transition_matrix(z[0], gamma)
        for s in range(2):
            alpha[0, s] = pi[s] * stats.norm.pdf(y[0], mu[s], sigma[s])

        c = alpha[0].sum()
        if c > 0:
            alpha[0] /= c
            log_likelihood += np.log(c)

        # t=1,...,T-1
        for t in range(1, T):
            P_t = self._time_varying_transition_matrix(z[t - 1], gamma)

            for s in range(2):
                alpha[t, s] = np.dot(alpha[t - 1], P_t[:, s]) * stats.norm.pdf(y[t], mu[s], sigma[s])

            c = alpha[t].sum()
            if c > 0:
                alpha[t] /= c
                log_likelihood += np.log(c)

        return -log_likelihood

    def _forward_backward_time_varying(self, y, z, params):
        """时变转换概率的前向-后向算法"""
        T = len(y)

        # 解包参数
        mu = params[0:2]
        sigma = np.abs(params[2:4])
        gamma = params[4:8]

        # 初始概率
        pi = np.array([0.5, 0.5])

        # 前向概率
        alpha = np.zeros((T, 2))
        c = np.zeros(T)

        # t=0
        P_0 = self._time_varying_transition_matrix(z[0], gamma)
        for s in range(2):
            alpha[0, s] = pi[s] * stats.norm.pdf(y[0], mu[s], sigma[s])
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0]

        # t=1,...,T-1
        for t in range(1, T):
            P_t = self._time_varying_transition_matrix(z[t - 1], gamma)

            for s in range(2):
                alpha[t, s] = np.dot(alpha[t - 1], P_t[:, s]) * stats.norm.pdf(y[t], mu[s], sigma[s])

            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]

        # 后向概率
        beta = np.zeros((T, 2))
        beta[T - 1] = 1

        for t in range(T - 2, -1, -1):
            P_t = self._time_varying_transition_matrix(z[t], gamma)

            for s in range(2):
                for j in range(2):
                    beta[t, s] += P_t[s, j] * stats.norm.pdf(y[t + 1], mu[j], sigma[j]) * beta[t + 1, j]

            if c[t + 1] > 0:
                beta[t] /= c[t + 1]

        # 平滑概率
        gamma_smooth = alpha * beta
        for t in range(T):
            if gamma_smooth[t].sum() > 0:
                gamma_smooth[t] /= gamma_smooth[t].sum()

        return gamma_smooth, alpha

    def fit(self, y, z, max_iter=100):
        """拟合时变马尔可夫转换模型"""
        # 标准化过渡变量
        z_std = (z - np.mean(z)) / np.std(z)

        # 初始参数
        mu_init = [np.percentile(y, 25), np.percentile(y, 75)]
        sigma_init = [np.std(y) * 0.5, np.std(y) * 1.5]
        gamma_init = [0, 0.5, 0, -0.5]

        params_init = mu_init + sigma_init + gamma_init

        # 参数边界
        bounds = [
            (None, None), (None, None),
            (0.001, None), (0.001, None),
            (None, None), (None, None), (None, None), (None, None)
        ]

        # 优化
        result = minimize(
            self._log_likelihood,
            params_init,
            args=(y, z_std),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )

        # 存储结果
        self.params = result.x
        self.transition_params = result.x[4:8]

        # 计算平滑概率
        self.smoothed_probs, self.filtered_probs = self._forward_backward_time_varying(y, z_std, self.params)

        return self

    def get_regime_probs(self, regime=1):
        """获取特定制度的概率"""
        return self.smoothed_probs[:, regime]


def plot_figure1_regime_transition(model1, tv_model):
    """
    绘制图1：制度概率和过渡变量
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 准备数据
    dates = pd.to_datetime(model1['DT'])
    years = dates.dt.year + dates.dt.dayofyear / 365.25
    transition_var = model1['BTC_lag1'].values
    regime_probs = tv_model.get_regime_probs(regime=1)

    # 左轴：过渡变量（虚线）
    ax1.set_xlabel('')
    ax1.set_ylabel('', fontsize=10)
    line1 = ax1.plot(years, transition_var, color='#404040', linestyle=':', linewidth=1.5, alpha=0.8)
    ax1.tick_params(axis='y', labelsize=9)

    # 设置y轴范围
    y_min = np.percentile(transition_var, 1)
    y_max = np.percentile(transition_var, 99)
    y_range = y_max - y_min
    ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # 右轴：制度概率（粗实线）
    ax2 = ax1.twinx()
    ax2.set_ylabel('', fontsize=10)
    line2 = ax2.plot(years, regime_probs, color='#404040', linewidth=2.5)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.set_ylim(-0.05, 1.05)

    # 设置x轴
    ax1.set_xlim(years.iloc[0], years.iloc[-1])

    # 添加网格
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)


    plt.tight_layout()
    return fig


def analyze_time_varying_markov(model1):
    """
    执行时变马尔可夫转换模型分析
    """
    print("正在拟合时变马尔可夫转换模型...")

    # 准备数据
    y = model1['RV'].values
    z = model1['BTC_lag1'].values  # 过渡变量

    # 创建并拟合模型
    tv_model = TimeVaryingMarkovSwitching(n_regimes=2)
    tv_model.fit(y, z, max_iter=100)

    # 获取结果
    regime_probs = tv_model.get_regime_probs(regime=1)

    # 打印统计信息
    print(f"\n模型参数:")
    print(f"状态0 (低波动): μ = {tv_model.params[0]:.4f}, σ = {tv_model.params[2]:.4f}")
    print(f"状态1 (高波动): μ = {tv_model.params[1]:.4f}, σ = {tv_model.params[3]:.4f}")
    print(f"\n转换概率参数 (γ): {tv_model.transition_params}")
    print(f"\n制度1的平均概率: {regime_probs.mean():.4f}")
    print(f"制度1的期数: {(regime_probs > 0.5).sum()} / {len(regime_probs)}")

    # 绘制图1
    print("\n绘制图1：制度概率和过渡变量...")
    fig1 = plot_figure1_regime_transition(model1, tv_model)
    plt.show()

    return tv_model, fig1


# 执行分析
if __name__ == "__main__":
    tv_model, fig1 = analyze_time_varying_markov(model1)

    # 如果需要保存图形
    fig1.savefig('cj.png', dpi=300, bbox_inches='tight')
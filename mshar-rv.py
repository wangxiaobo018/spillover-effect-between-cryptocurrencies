import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings


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


import os
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv
import os


# Read the data
df_data = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv")
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

# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
from datetime import date

# Convert DT to datetime.date
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

# 正确计算滞后特征
rv_lag1 = RV['RV'].shift(1)   # 滞后1期
rv_lag5 = RV['RV'].rolling(window=5).mean().shift(1)
rv_lag22 = RV['RV'].rolling(window=22).mean().shift(1)

data_rv = pd.DataFrame({
    'RV': RV['RV'],
    'rv_lag1': rv_lag1,
    'rv_lag5': rv_lag5,
    'rv_lag22': rv_lag22
})

data_rv = data_rv.dropna()
print(data_rv)

test_size = 300
# 划分训练集和测试集
train_data = data_rv.iloc[:len(data_rv) - test_size]
test_data = data_rv.iloc[len(data_rv) - test_size:]


import os
import json
from scipy.stats import norm
from scipy.special import logsumexp
from numpy.linalg import inv


def forward_backward(y, X, betas, sigmas, P, pi):
    """
    前向-后向算法实现，计算状态概率和似然

    参数:
    y: 因变量
    X: 特征矩阵
    betas: 各状态的系数
    sigmas: 各状态的波动率
    P: 状态转移概率矩阵
    pi: 初始状态概率

    返回:
    gamma: 状态概率
    xi: 转移概率
    likelihood: 似然值
    """
    T = len(y)
    K = len(sigmas)

    log_lik = np.zeros((T, K))
    for k in range(K):
        residuals = y - X @ betas[k]
        log_lik[:, k] = norm.logpdf(residuals, scale=sigmas[k])

    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi + 1e-10) + log_lik[0]

    log_P = np.log(P + 1e-10)

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t - 1] + log_P[:, j]) + log_lik[t, j]

    log_beta = np.zeros((T, K))

    for t in range(T - 2, -1, -1):
        for i in range(K):
            terms = log_P[i, :] + log_lik[t + 1, :] + log_beta[t + 1, :]
            log_beta[t, i] = logsumexp(terms)

    log_gamma = log_alpha + log_beta
    for t in range(T):
        log_gamma[t] -= logsumexp(log_gamma[t])
    gamma = np.exp(log_gamma)

    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (log_alpha[t, i] +
                               log_P[i, j] +
                               log_lik[t + 1, j] +
                               log_beta[t + 1, j])
        xi[t] = np.exp(xi[t] - logsumexp(xi[t]))

    likelihood = np.exp(logsumexp(log_alpha[-1]))
    return gamma, xi, likelihood


def initialize_parameters(K, p):
    """
    初始化模型参数

    参数:
    K: 状态数
    p: 参数数

    返回:
    betas, sigmas, P, pi: 初始化的参数
    """
    betas = np.zeros((K, p))
    # 为不同状态设置不同的初始值
    betas[:, 0] = [0.01, 0.02]
    betas[:, 1:] = np.array([[0.4, 0.3, 0.2], [0.3, 0.4, 0.2]])
    sigmas = np.array([0.01, 0.02])
    # 高持续概率的转移矩阵
    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    return betas, sigmas, P, pi


def calculate_standard_errors(X, y, betas, sigmas, gamma, K, p):
    """
    计算参数标准误

    参数:
    X: 特征矩阵
    y: 因变量
    betas: 各状态的系数
    sigmas: 各状态的波动率
    gamma: 状态概率
    K: 状态数
    p: 参数数

    返回:
    beta_se: 系数标准误
    sigma_se: 波动率标准误
    """
    T = len(y)
    beta_se = np.zeros_like(betas)
    sigma_se = np.zeros_like(sigmas)

    for k in range(K):
        # 为每个状态计算加权最小二乘的标准误
        W = np.diag(gamma[:, k])
        XtWX = X.T @ W @ X

        # 添加微小的正则化项以确保矩阵可逆
        XtWX_inv = inv(XtWX + 1e-8 * np.eye(p))

        # 计算残差
        residuals = y - X @ betas[k]
        weighted_residuals_sq = gamma[:, k] * (residuals ** 2)
        weighted_sum = np.sum(gamma[:, k])

        # 计算系数标准误
        mse = np.sum(weighted_residuals_sq) / (weighted_sum - p)
        beta_se[k] = np.sqrt(np.diag(XtWX_inv) * mse)

        # 计算sigma的标准误
        # 根据ML估计的标准误近似公式
        sigma_se[k] = sigmas[k] / np.sqrt(2 * weighted_sum)

    return beta_se, sigma_se


def calculate_transition_prob_se(xi, gamma, T):
    """
    计算转移概率矩阵的标准误

    参数:
    xi: 状态转移概率
    gamma: 状态概率
    T: 观测数量

    返回:
    P_se: 转移概率标准误矩阵
    """
    K = gamma.shape[1]
    P_se = np.zeros((K, K))

    for i in range(K):
        state_count = np.sum(gamma[:-1, i])
        for j in range(K):
            transition_count = np.sum(xi[:, i, j])
            p_ij = transition_count / state_count if state_count > 0 else 0
            # 二项分布标准误
            P_se[i, j] = np.sqrt(p_ij * (1 - p_ij) / (state_count + 1e-10))

    return P_se


def em_mshar(data_rv, max_iter=100, tol=1e-6):
    """
    使用EM算法估计马尔可夫切换HAR-RV模型的参数

    参数:
    data_rv: 数据
    max_iter: 最大迭代次数
    tol: 收敛容差

    返回:
    betas, sigmas, P, pi, gamma, model_info: 估计的参数和模型信息
    """
    X, y = prepare_X_y(data_rv)
    T, p = X.shape
    K = 2

    betas, sigmas, P, pi = initialize_parameters(K, p)
    old_likelihood = -np.inf

    converged = False
    iteration = 0

    for iteration in range(max_iter):
        # E步：计算状态概率
        gamma, xi, likelihood = forward_backward(y, X, betas, sigmas, P, pi)

        # M步：更新参数
        for k in range(K):
            W = np.diag(gamma[:, k] + 1e-10)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y

            # 增加正则化以提高稳定性
            ridge = 1e-6 * np.eye(p)
            try:
                betas[k] = np.linalg.solve(XtWX + ridge, XtWy)
            except np.linalg.LinAlgError:
                # 如果矩阵接近奇异，增加正则化
                ridge = 1e-4 * np.eye(p)
                betas[k] = np.linalg.solve(XtWX + ridge, XtWy)

            residuals = y - X @ betas[k]
            weighted_sum = np.sum(gamma[:, k])
            if weighted_sum > 1e-10:
                sigmas[k] = np.sqrt(np.sum(gamma[:, k] * residuals ** 2) / weighted_sum)
            else:
                sigmas[k] = 0.01

        # 更新转移概率
        P = xi.sum(axis=0)
        row_sums = P.sum(axis=1, keepdims=True)
        P = (P + 1e-10) / (row_sums + K * 1e-10)

        # 更新初始状态概率
        pi = gamma[0]

        # 检查收敛
        if abs(np.log(likelihood) - old_likelihood) < tol and iteration > 0:
            converged = True
            break

        old_likelihood = np.log(likelihood)

    # 计算标准误
    beta_se, sigma_se = calculate_standard_errors(X, y, betas, sigmas, gamma, K, p)
    P_se = calculate_transition_prob_se(xi, gamma, T)

    # 计算t统计量和p值
    t_stats = np.zeros_like(betas)
    p_values = np.zeros_like(betas)

    for k in range(K):
        t_stats[k] = betas[k] / beta_se[k]
        # 双尾t检验的p值
        p_values[k] = 2 * (1 - norm.cdf(np.abs(t_stats[k])))

    # 收集模型信息
    model_info = {
        'betas': betas,
        'beta_se': beta_se,
        't_stats': t_stats,
        'p_values': p_values,
        'sigmas': sigmas,
        'sigma_se': sigma_se,
        'P': P,
        'P_se': P_se,
        'pi': pi,
        'likelihood': likelihood,
        'iterations': iteration + 1,
        'converged': converged,
        'AIC': -2 * np.log(likelihood) + 2 * (2 * p + 2 + 2),  # 参数数量: 2状态*p系数 + 2个sigma + 2个独立转移概率
        'BIC': -2 * np.log(likelihood) + np.log(T) * (2 * p + 2 + 2)
    }

    return betas, sigmas, P, pi, gamma, model_info


def prepare_X_y(data_rv):
    """
    准备特征矩阵和因变量

    参数:
    data_rv: 数据

    返回:
    X, y: 特征矩阵和因变量
    """
    X = np.column_stack([
        np.ones(len(data_rv)),
        data_rv['rv_lag1'].values,
        data_rv['rv_lag5'].values,
        data_rv['rv_lag22'].values
    ])
    y = data_rv['RV'].values
    return X, y


def prepare_step_features(data_subset):
    """
    为预测准备特征矩阵

    参数:
    data_subset: 数据子集

    返回:
    X: 特征矩阵
    """
    X = np.column_stack([
        np.ones(len(data_subset)),
        data_subset['rv_lag1'].values,
        data_subset['rv_lag5'].values,
        data_subset['rv_lag22'].values
    ])
    return X


def mshar_predict_1step(X, betas, sigmas, P, gamma_last):
    """
    使用马尔可夫切换HAR-RV模型进行一步预测

    参数:
    X: 输入特征向量 (单个观测)
    betas: 各状态的系数
    sigmas: 各状态的波动率
    P: 状态转移概率矩阵
    gamma_last: 最后一个观测值的状态概率

    返回:
    prediction: 预测值
    """
    K = len(sigmas)

    # 计算各状态下的预测值
    regime_predictions = np.array([X @ betas[k] for k in range(K)])

    # 加权平均得到最终预测
    prediction = np.sum(regime_predictions * gamma_last)

    return prediction

def rolling_window_mshar_prediction(train_data, test_data, horizons=[1, 5, 22], window_size=None):
    """
    使用滚动窗口进行马尔可夫切换HAR-RV模型预测，匹配所要求的预测逻辑，去掉指数化操作

    参数:
    train_data: 训练数据（log(RV) 形式）
    test_data: 测试数据（log(RV) 形式）
    horizons: 预测步长列表
    window_size: 滚动窗口大小

    返回:
    predictions_dict, actuals_dict, first_step_params: 预测结果，实际值和首次估计的参数
    """
    predictions_dict = {h: [] for h in horizons}
    actuals_dict = {h: [] for h in horizons}

    # 用于存储第一步估计的参数
    first_step_params = None

    if window_size is None:
        window_size = len(train_data)

    rolling_data = train_data.copy()
    n_test_points = len(test_data)

    for i in range(n_test_points):
        if i % 50 == 0:
            print(f"处理测试点: {i + 1}/{n_test_points}")

        try:
            # 在当前窗口上拟合模型
            betas, sigmas, P, pi, gamma, model_info = em_mshar(rolling_data)

            # 保存第一步估计的参数并打印
            if i == 0:
                first_step_params = model_info

                # 打印第一步的参数和标准误
                print("\n===== 第一步模型参数 =====")
                for k in range(2):
                    print(f"状态{k + 1}系数:")
                    param_names = ['常数项', 'RV_t-1', 'RV_t-5', 'RV_t-22']
                    for j in range(len(param_names)):
                        print(
                            f"  {param_names[j]}: {betas[k][j]:.6f} (标准误: {model_info['beta_se'][k][j]:.6f}, t值: {model_info['t_stats'][k][j]:.4f}, p值: {model_info['p_values'][k][j]:.4f})")
                    print(f"状态{k + 1}波动率: {sigmas[k]:.6f} (标准误: {model_info['sigma_se'][k]:.6f})")

                print(f"转移概率矩阵:")
                print(f"  P00: {P[0][0]:.4f} (标准误: {model_info['P_se'][0][0]:.4f})")
                print(f"  P11: {P[1][1]:.4f} (标准误: {model_info['P_se'][1][1]:.4f})")
                print(f"初始状态概率: [{pi[0]:.4f}, {pi[1]:.4f}]")
                print(f"对数似然值: {np.log(model_info['likelihood']):.4f}")
                print(f"AIC: {model_info['AIC']:.4f}")
                print(f"BIC: {model_info['BIC']:.4f}")
                print("=========================\n")

            # 一步预测
            X_1step = prepare_step_features(test_data.iloc[i:i + 1])
            pred_1 = mshar_predict_1step(X_1step[0], betas, sigmas, P, gamma[-1])
            predictions_dict[1].append(pred_1)
            actuals_dict[1].append(test_data['RV'].iloc[i])  # 直接使用 log(RV)

            # 五步预测 - 直接预测t+4时刻
            if i + 4 < n_test_points:
                X_5step = prepare_step_features(test_data.iloc[i + 4:i + 5])
                pred_5 = mshar_predict_1step(X_5step[0], betas, sigmas, P, gamma[-1])
                predictions_dict[5].append(pred_5)
                actuals_dict[5].append(test_data['RV'].iloc[i + 4])  # 直接使用 log(RV)
            else:
                predictions_dict[5].append(None)
                actuals_dict[5].append(None)

            # 22步预测 - 直接预测t+21时刻
            if i + 21 < n_test_points:
                X_22step = prepare_step_features(test_data.iloc[i + 21:i + 22])
                pred_22 = mshar_predict_1step(X_22step[0], betas, sigmas, P, gamma[-1])
                predictions_dict[22].append(pred_22)
                actuals_dict[22].append(test_data['RV'].iloc[i + 21])  # 直接使用 log(RV)
            else:
                predictions_dict[22].append(None)
                actuals_dict[22].append(None)

        except Exception as e:
            print(f"预测点 {i} 发生错误: {str(e)}")
            for h in horizons:
                predictions_dict[h].append(None)
                if i + h - 1 < n_test_points:
                    actuals_dict[h].append(test_data['RV'].iloc[i + h - 1])  # 直接使用 log(RV)
                else:
                    actuals_dict[h].append(None)

        # 更新滚动窗口
        if len(rolling_data) >= window_size:
            rolling_data = rolling_data.iloc[1:].copy()

        if i < n_test_points:
            new_point = test_data.iloc[i:i + 1].copy()
            rolling_data = pd.concat([rolling_data, new_point])

    return predictions_dict, actuals_dict, first_step_params

def run_mshar_forecast(train_data, test_data, horizons=[1, 5, 22], window_size=None,
                       output_dir='mshar_results'):
    """
    运行马尔可夫切换HAR-RV模型的预测，仅保存预测值，打印第一次估计结果

    参数:
    train_data: 训练数据
    test_data: 测试数据
    horizons: 预测步长列表
    window_size: 滚动窗口大小
    output_dir: 结果保存目录

    返回:
    results_dict: 包含预测结果和实际值的字典
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 执行滚动窗口预测
    print("开始滚动窗口预测...")
    predictions_dict, actuals_dict, _ = rolling_window_mshar_prediction(
        train_data, test_data, horizons, window_size
    )
    print("预测完成")

    # 初始化结果字典，仅包含预测和实际值
    results_dict = {'predictions': {}, 'actuals': {}}

    # 保存每个预测步长的结果
    for h in horizons:
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            f'预测_RV_{h}步': predictions_dict[h],
            f'实际_RV_{h}步': actuals_dict[h]
        })

        # 添加日期索引
        if hasattr(test_data, 'index') and len(test_data.index) >= len(results_df):
            results_df.index = test_data.index[:len(results_df)]

        # 保存结果
        h_save_path = os.path.join(output_dir, f'mshar_forecast_h{h}.csv')
        results_df.to_csv(h_save_path)
        print(f"已保存 {h} 步预测结果")

        # 保存到结果字典
        results_dict['predictions'][h] = predictions_dict[h]
        results_dict['actuals'][h] = actuals_dict[h]

    # 保存所有预测结果到一个文件
    all_results = pd.DataFrame()

    for h in horizons:
        all_results[f'预测_RV_{h}步'] = predictions_dict[h]
        all_results[f'实际_RV_{h}步'] = actuals_dict[h]

    # 添加日期索引
    if hasattr(test_data, 'index') and len(test_data.index) >= len(all_results):
        all_results.index = test_data.index[:len(all_results)]

    # 保存所有结果
    all_save_path = os.path.join(output_dir, 'mshar_forecast_all_horizons.csv')
    all_results.to_csv(all_save_path)
    print(f"已保存所有预测步长的结果")

    return results_dict


# 示例使用方法
def run_model_example():
    """
    运行模型示例
    注意: 需要提前准备好train_data和test_data
    train_data和test_data必须包含'RV', 'rv_lag1', 'rv_lag5', 'rv_lag22'列
    """
    # 设置输出文件夹
    output_dir = 'mshar_results'

    # 运行模型
    results = run_mshar_forecast(
        train_data=train_data,
        test_data=test_data,
        horizons=[1, 5, 22],  # 您可以指定需要的预测步长
        window_size=None,  # 滚动窗口大小
        output_dir=output_dir  # 结果保存目录
    )

    return results


# 如果作为主程序运行
if __name__ == "__main__":
    # 运行模型
    run_model_example()




import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings


import numpy as np
import pandas as pd
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


# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")

data_filtered = df[df['code'] == "BTC"].copy()
# 按组进行分类统计
group_summary = df.groupby('code').size().reset_index(name='NumObservations')

def get_re(data, alpha):
    # 将数据转换为DataFrame并确保是副本
    result = data.copy()

    # 转换时间列 - 使用更健壮的方式处理日期
    try:
        # 如果时间格式是 "YYYY/M/D H" 这种格式
        result['day'] = pd.to_datetime(result['time'], format='%Y/%m/%d %H')
    except:
        try:
            # 如果上面的格式不工作，尝试其他常见格式
            result['day'] = pd.to_datetime(result['time'])
        except:
            # 如果还是不行，尝试先分割时间字符串
            result['day'] = pd.to_datetime(result['time'].str.split().str[0])

    # 只保留日期部分
    result['day'] = result['day'].dt.date

    # 按天分组进行计算
    def calculate_daily_metrics(group):
        # 计算简单收益率
        group['Ret'] = (group['close'] / group['close'].shift(1) - 1) * 100
        group['Ret'].iloc[0] = 0  # First return is 0
        group['Ret'][group['close'].shift(1) == 0] = np.nan  # Handle division by zero

        # 删除缺失值
        group = group.dropna()

        if len(group) == 0:
            return None

        # 计算标准差
        sigma = group['Ret'].std()

        # 计算分位数阈值
        r_minus = norm.ppf(alpha) * sigma
        r_plus = norm.ppf(1 - alpha) * sigma

        # 计算超额收益
        REX_minus = np.sum(np.where(group['Ret'] <= r_minus, group['Ret'] ** 2, 0))
        REX_plus = np.sum(np.where(group['Ret'] >= r_plus, group['Ret'] ** 2, 0))
        REX_moderate = np.sum(np.where((group['Ret'] > r_minus) & (group['Ret'] < r_plus),
                                       group['Ret'] ** 2, 0))

        return pd.Series({
            'REX_minus': REX_minus,
            'REX_plus': REX_plus,
            'REX_moderate': REX_moderate
        })

    # 按天分组计算指标
    result = result.groupby('day').apply(calculate_daily_metrics).reset_index()

    return result

# 使用函数
har_re = get_re(data_filtered, alpha=0.05)
print(har_re)


# Create data_ret DataFrame with renamed columns first
data_ret = df[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()

# Calculate returns for each group using the new formula
def calculate_returns(prices):
    # Compute daily returns using the given formula
    returns = (prices / prices.shift(1) - 1) * 100  # (Pt - Pt-1) / Pt-1 * 100
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
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']

# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])


# Ensure 'har_cj' is properly prepared before merging
har_pd_re = pd.merge(RV, har_re, left_index=True, right_index=True)


har_pd_re['rex_m_lag1'] = har_pd_re['REX_minus'].shift(1)
har_pd_re['rex_m_lag5'] = har_pd_re['REX_minus'].rolling(window=5).mean().shift(1)
har_pd_re['rex_m_lag22'] = har_pd_re['REX_minus'].rolling(window=22).mean().shift(1)

har_pd_re['rex_p_lag1'] = har_pd_re['REX_plus'].shift(1)
har_pd_re['rex_p_lag5'] = har_pd_re['REX_plus'].rolling(window=5,).mean().shift(1)
har_pd_re['rex_p_lag22'] = har_pd_re['REX_plus'].rolling(window=22).mean().shift(1)
# #
har_pd_re['rex_moderate_lag1'] = har_pd_re['REX_moderate'].shift(1)
har_pd_re['rex_moderate_lag5'] = har_pd_re['REX_moderate'].rolling(window=5).mean().shift(1)
har_pd_re['rex_moderate_lag22'] = har_pd_re['REX_moderate'].rolling(window=22).mean().shift(1)


model_data = har_pd_re[['RV', 'rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                       'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                      'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']]
# #
# # # 删除缺失值
model_data = model_data.dropna()
print(model_data)

test_size = 300
# 划分训练集和测试集
train_data = model_data.iloc[:len(model_data) - test_size]
test_data = model_data.iloc[len(model_data) - test_size:]


import pandas as pd
import numpy as np
import os
import json
from scipy.stats import norm
from scipy.special import logsumexp
from numpy.linalg import inv

import os
import json

from scipy.stats import norm
from scipy.special import logsumexp
from numpy.linalg import inv


# MODIFIED forward_backward
def forward_backward(y, X, betas, sigmas, P, pi):
    """
    前向-后向算法实现，返回对数似然
    """
    T = len(y)
    K = len(sigmas)

    log_lik = np.zeros((T, K))
    for k in range(K):
        # <--- Safety check for very small sigmas --->
        safe_sigma = np.maximum(sigmas[k], 1e-8)
        residuals = y - X @ betas[k]
        log_lik[:, k] = norm.logpdf(residuals, scale=safe_sigma)

    # --- The rest of the algorithm is numerically stable and remains the same ---
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
                xi[t, i, j] = (log_alpha[t, i] + log_P[i, j] +
                               log_lik[t + 1, j] + log_beta[t + 1, j])
        xi[t] = np.exp(xi[t] - logsumexp(xi[t]))

    # <--- Return log-likelihood directly --->
    log_likelihood = logsumexp(log_alpha[-1])
    return gamma, xi, log_likelihood


# MODIFIED initialize_parameters
# 修改后的 initialize_parameters (不再打印)
def initialize_parameters(K, p, X, y, verbose=True):
    """
    初始化模型参数，sigmas根据数据的初步OLS回归结果动态生成
    """
    # ... (betas的初始化代码保持不变) ...
    betas = np.zeros((K, p))
    betas[:, 0] = [0.01, 0.02]
    betas[0, 1:4] = [0.2, 0.15, 0.1]
    betas[1, 1:4] = [0.3, 0.2, 0.15]
    betas[0, 4:7] = [0.2, 0.15, 0.1]
    betas[1, 4:7] = [0.3, 0.2, 0.15]
    betas[0, 7:10] = [0.2, 0.15, 0.1]
    betas[1, 7:10] = [0.3, 0.2, 0.15]

    try:
        # Simple OLS to estimate the scale of residuals
        ols_beta = np.linalg.solve(X.T @ X + 1e-6 * np.eye(p), X.T @ y)
        ols_residuals = y - X @ ols_beta
        std_resid = np.std(ols_residuals)

        # Set different state sigmas based on the baseline
        sigmas = np.array([0.75 * std_resid, 1.25 * std_resid])

        # --- 下面这行就是你想去掉的打印语句，在前面加上 # ---
        # if verbose:
        #     print(f"根据数据动态初始化: 估计的残差标准差 = {std_resid:.4f}, 初始sigmas = [{sigmas[0]:.4f}, {sigmas[1]:.4f}]")

    except np.linalg.LinAlgError:
        # --- 同样注释掉失败时的打印信息 ---
        # if verbose:
        #     print("OLS初始化失败，使用默认sigma值")
        sigmas = np.array([0.5, 0.8])

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


def prepare_X_y(model_data):
    """
    准备特征矩阵和因变量，针对MS-HAR-RE模型

    参数:
    model_data: 包含RV和RE组件的数据框

    返回:
    X, y: 特征矩阵和因变量
    """
    X = np.column_stack([
        np.ones(len(model_data)),
        # 负极值收益
        model_data['rex_m_lag1'].values,
        model_data['rex_m_lag5'].values,
        model_data['rex_m_lag22'].values,
        # 正极值收益
        model_data['rex_p_lag1'].values,
        model_data['rex_p_lag5'].values,
        model_data['rex_p_lag22'].values,
        # 中等波动
        model_data['rex_moderate_lag1'].values,
        model_data['rex_moderate_lag5'].values,
        model_data['rex_moderate_lag22'].values
    ])
    y = model_data['RV'].values
    return X, y


# MODIFIED em_mshar
def em_mshar(model_data, max_iter=100, tol=1e-6, verbose=True):
    """
    使用EM算法估计马尔可夫切换HAR-RE模型的参数
    """
    X, y = prepare_X_y(model_data)
    T, p = X.shape
    K = 2

    # <--- Modified initialization call --->
    betas, sigmas, P, pi = initialize_parameters(K, p, X, y, verbose=verbose)

    # <--- Use log-likelihood for tracking convergence --->
    old_log_likelihood = -np.inf
    converged = False
    iteration = 0

    for iteration in range(max_iter):
        # E-step: now receives log_likelihood
        gamma, xi, log_likelihood = forward_backward(y, X, betas, sigmas, P, pi)

        # M-step: remains the same logic, but with a sigma safety check
        for k in range(K):
            W = np.diag(gamma[:, k] + 1e-10)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            ridge = 1e-6 * np.eye(p)
            try:
                betas[k] = np.linalg.solve(XtWX + ridge, XtWy)
            except np.linalg.LinAlgError:
                ridge = 1e-4 * np.eye(p)
                betas[k] = np.linalg.solve(XtWX + ridge, XtWy)

            residuals = y - X @ betas[k]
            weighted_sum = np.sum(gamma[:, k])
            if weighted_sum > 1e-10:
                sigmas[k] = np.sqrt(np.sum(gamma[:, k] * residuals ** 2) / weighted_sum)
                if sigmas[k] < 1e-8:  # Safety check
                    sigmas[k] = 1e-8
            else:
                sigmas[k] = 0.01

        P = xi.sum(axis=0)
        row_sums = P.sum(axis=1, keepdims=True)
        P = (P + 1e-10) / (row_sums + K * 1e-10)
        pi = gamma[0]

        # <--- Convergence check using log-likelihood --->
        if not np.isfinite(log_likelihood):
            if verbose: print(f"Iteration {iteration}: Log-likelihood became non-finite. Stopping.")
            break
        if abs(log_likelihood - old_log_likelihood) < tol and iteration > 0:
            converged = True
            break

        old_log_likelihood = log_likelihood

    # ... (Standard error calculations remain the same) ...
    beta_se, sigma_se = calculate_standard_errors(X, y, betas, sigmas, gamma, K, p)
    P_se = calculate_transition_prob_se(xi, gamma, T)
    t_stats = betas / (beta_se + 1e-10)
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

    # <--- Modified model_info dictionary --->
    num_params = K * p + K + K * (K - 1)
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
        'log_likelihood': log_likelihood,  # Store log_likelihood directly
        'iterations': iteration + 1,
        'converged': converged,
        'AIC': -2 * log_likelihood + 2 * num_params,
        'BIC': -2 * log_likelihood + np.log(T) * num_params
    }

    return betas, sigmas, P, pi, gamma, model_info

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
        # 负极值收益
        data_subset['rex_m_lag1'].values,
        data_subset['rex_m_lag5'].values,
        data_subset['rex_m_lag22'].values,
        # 正极值收益
        data_subset['rex_p_lag1'].values,
        data_subset['rex_p_lag5'].values,
        data_subset['rex_p_lag22'].values,
        # 中等波动
        data_subset['rex_moderate_lag1'].values,
        data_subset['rex_moderate_lag5'].values,
        data_subset['rex_moderate_lag22'].values
    ])
    return X


def mshar_predict_1step(X, betas, sigmas, P, gamma_last):
    """
    使用马尔可夫切换HAR-RE模型进行一步预测

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
    使用滚动窗口进行马尔可夫切换HAR-RV模型预测，匹配所要求的预测逻辑

    参数:
    train_data: 训练数据
    test_data: 测试数据
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
                    param_names = [
                        '常数项',
                        'rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                        'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                        'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22'
                    ]

                    for k in range(2):
                        print(f"状态{k + 1}系数:")
                        # 这个循环现在会打印出所有10个参数，并使用正确的标签
                        for j in range(len(param_names)):
                            print(
                                f"  {param_names[j]}: {betas[k][j]:.6f} "
                                f"(标准误: {model_info['beta_se'][k][j]:.6f}, "
                                f"t值: {model_info['t_stats'][k][j]:.4f}, "
                                f"p值: {model_info['p_values'][k][j]:.4f})")
                        print(f"状态{k + 1}波动率: {sigmas[k]:.6f} (标准误: {model_info['sigma_se'][k]:.6f})")

                    print(f"转移概率矩阵:")
                    print(f"  P00: {P[0][0]:.4f} (标准误: {model_info['P_se'][0][0]:.4f})")
                    print(f"  P11: {P[1][1]:.4f} (标准误: {model_info['P_se'][1][1]:.4f})")
                    print(f"初始状态概率: [{pi[0]:.4f}, {pi[1]:.4f}]")
                    print(f"对数似然值: {model_info['log_likelihood']:.4f}")
                    print(f"AIC: {model_info['AIC']:.4f}")
                    print(f"BIC: {model_info['BIC']:.4f}")
                    print("=========================================\n")

            # 一步预测
            X_1step = prepare_step_features(test_data.iloc[i:i + 1])
            pred_1 = mshar_predict_1step(X_1step[0], betas, sigmas, P, gamma[-1])

            predictions_dict[1].append(pred_1)
            actuals_dict[1].append(test_data['RV'].iloc[i])  # 实际值也要指数还原

            # 五步预测 - 直接预测t+4时刻
            if i + 4 < n_test_points:
                X_5step = prepare_step_features(test_data.iloc[i + 4:i + 5])
                pred_5 = mshar_predict_1step(X_5step[0], betas, sigmas, P, gamma[-1])

                predictions_dict[5].append(pred_5)
                actuals_dict[5].append(test_data['RV'].iloc[i + 4]) # 实际值也要指数还原
            else:
                predictions_dict[5].append(None)
                actuals_dict[5].append(None)

            # 22步预测 - 直接预测t+21时刻
            if i + 21 < n_test_points:
                X_22step = prepare_step_features(test_data.iloc[i + 21:i + 22])
                pred_22 = mshar_predict_1step(X_22step[0], betas, sigmas, P, gamma[-1])
                # 直接指数还原
                predictions_dict[22].append(pred_22)
                actuals_dict[22].append(test_data['RV'].iloc[i + 21])  # 实际值也要指数还原
            else:
                predictions_dict[22].append(None)
                actuals_dict[22].append(None)

        except Exception as e:
            print(f"预测点 {i} 发生错误: {str(e)}")
            for h in horizons:
                predictions_dict[h].append(None)
                if i + h - 1 < n_test_points:
                    actuals_dict[h].append(np.exp(test_data['RV'].iloc[i + h - 1]))  # 实际值也要指数还原
                else:
                    actuals_dict[h].append(None)

        # 更新滚动窗口
        if len(rolling_data) >= window_size:
            rolling_data = rolling_data.iloc[1:].copy()

        if i < n_test_points:
            new_point = test_data.iloc[i:i + 1].copy()
            rolling_data = pd.concat([rolling_data, new_point])

    return predictions_dict, actuals_dict, first_step_params


def run_mshar_re_forecast(train_data, test_data, horizons=[1, 5, 22], window_size=None,
                          output_dir='mshar_re_results'):
    """
    运行马尔可夫切换HAR-RE模型的预测

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
    predictions_dict, actuals_dict, first_step_params = rolling_window_mshar_prediction(
        train_data, test_data, horizons, window_size
    )
    print("预测完成")

    # 初始化结果字典
    results_dict = {'predictions': {}, 'actuals': {}, 'first_step_params': first_step_params}

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
        h_save_path = os.path.join(output_dir, f'mshar_re_forecast_h{h}.csv')
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
    all_save_path = os.path.join(output_dir, 'mshar_re_forecast_all_horizons.csv')
    all_results.to_csv(all_save_path)
    print(f"已保存所有预测步长的结果")

    # 保存第一步估计的参数
    param_names = ['常数项',
                   'rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                   'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                   'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']
    parameter_info = {
        '模型信息': {
            '对数似然值': float(np.log(first_step_params['likelihood'])),
            'AIC': float(first_step_params['AIC']),
            'BIC': float(first_step_params['BIC']),
            '迭代次数': int(first_step_params['iterations']),
            '是否收敛': bool(first_step_params['converged'])
        }
    }

    # 为每个状态添加系数和标准误
    for k in range(2):
        state_info = {}
        # 添加系数和标准误
        for j, name in enumerate(param_names):
            state_info[f'{name}_系数'] = float(first_step_params['betas'][k][j])
            state_info[f'{name}_标准误'] = float(first_step_params['beta_se'][k][j])
            state_info[f'{name}_t值'] = float(first_step_params['t_stats'][k][j])
            state_info[f'{name}_p值'] = float(first_step_params['p_values'][k][j])

        # 添加波动率和标准误
        state_info['波动率'] = float(first_step_params['sigmas'][k])
        state_info['波动率_标准误'] = float(first_step_params['sigma_se'][k])

        parameter_info[f'状态{k + 1}'] = state_info

    # 添加转移概率和标准误
    parameter_info['转移概率'] = {
        'P00': float(first_step_params['P'][0][0]),
        'P00_标准误': float(first_step_params['P_se'][0][0]),
        'P11': float(first_step_params['P'][1][1]),
        'P11_标准误': float(first_step_params['P_se'][1][1]),
    }

    # 保存为JSON格式
    params_path = os.path.join(output_dir, 'mshar_re_first_step_parameters.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(parameter_info, f, indent=4, ensure_ascii=False)
    print(f"已保存第一步模型参数到: {params_path}")

    # 保存为CSV格式以便查看
    # 展平参数字典为CSV格式
    param_rows = []

    # 模型信息
    for key, value in parameter_info['模型信息'].items():
        param_rows.append({'参数': key, '值': value})

    # 状态参数
    for k in range(2):
        state_key = f'状态{k + 1}'
        for param, value in parameter_info[state_key].items():
            param_rows.append({'参数': f'{state_key}_{param}', '值': value})

    # 转移概率
    for param, value in parameter_info['转移概率'].items():
        param_rows.append({'参数': param, '值': value})

    param_df = pd.DataFrame(param_rows)
    param_csv_path = os.path.join(output_dir, 'mshar_re_first_step_parameters.csv')
    param_df.to_csv(param_csv_path, index=False, encoding='utf-8')
    print(f"已保存第一步模型参数到CSV: {param_csv_path}")

    return results_dict


# 示例使用方法
def run_model_example():
    """
    运行模型示例
    注意: 需要提前准备好train_data和test_data
    train_data和test_data必须包含'RV', 'rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                        'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                        'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22'列
    """
    # 设置输出文件夹
    output_dir = 'mshar_re_results'

    # 运行模型
    results = run_mshar_re_forecast(
        train_data=train_data,
        test_data=test_data,
        horizons=[1, 5, 22],  # 您可以指定需要的预测步长
        window_size=None,  # 滚动窗口大小
        output_dir=output_dir  # 结果保存目录
    )

    return results


# 如果作为主程序运行
if __name__ == "__main__":
    # 运行模型示例
    run_model_example()

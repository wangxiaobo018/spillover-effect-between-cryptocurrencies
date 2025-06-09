
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


# Read the data

# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")



data_filtered = df[df['code'] == "BTC"].copy()


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

        # 计算RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())


        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))


        term1 = abs_ret.iloc[4:].values  # Ret[5:n()]
        term2 = abs_ret.iloc[2:-2].values  # Ret[3:(n-2)]
        term3 = abs_ret.iloc[:-4].values  # Ret[1:(n-4)]

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

        # calculate JV
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({

            'BV': BV,
            'JV': JV,
            'C_t': C_t
        })


    result_df = pd.DataFrame(results)
    return result_df[['BV', 'JV', 'C_t']]

har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)
print(har_cj)


# Read the data

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


data_get_cj = pd.merge(RV, har_cj, left_index=True, right_index=True)

print(data_get_cj)

JV_lag1 = data_get_cj['JV'].shift(1)
C_t_lag1 = data_get_cj['C_t'].shift(1)
JV_lag5 = data_get_cj['JV'].rolling(window=5).mean().shift(1)
C_t_lag5 = data_get_cj['C_t'].rolling(window=5).mean().shift(1)
JV_lag22 = data_get_cj['JV'].rolling(window=22).mean().shift(1)
C_t_lag22 = data_get_cj['C_t'].rolling(window=22).mean().shift(1)

model_data= pd.DataFrame({
    'RV':data_get_cj['RV'],
    'Jv_lag1': JV_lag1,
    'Jv_lag5': JV_lag5,
    'Jv_lag22': JV_lag22,
    'C_t_lag1': C_t_lag1,
    'C_t_lag5': C_t_lag5,
    'C_t_lag22': C_t_lag22
})
model_data = model_data.dropna()
print(model_data)

test_size = 300
# 划分训练集和测试集
train_data = model_data.iloc[:len(model_data) - test_size]
test_data = model_data.iloc[len(model_data) - test_size:]



import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from numpy.linalg import inv
import os
import json


# 修改 initialize_parameters
def initialize_parameters(K, p, X, y, verbose=True):
    """
    初始化模型参数，sigmas根据数据的初步OLS回归结果动态生成

    参数:
    K: 状态数
    p: 参数数 (包括常数项)
    X: 特征矩阵
    y: 因变量
    verbose: 是否打印初始化信息

    返回:
    betas, sigmas, P, pi: 初始化的参数
    """
    # Beta系数的初始化，你的设置是正确的，我们保留它
    betas = np.zeros((K, p))
    betas[:, 0] = [0.01, 0.02]  # 常数项
    betas[:, 1:] = np.array([
        [0.3, 0.2, 0.1, 0.3, 0.2, 0.1],  # 状态1
        [0.2, 0.3, 0.1, 0.2, 0.3, 0.1]  # 状态2
    ])

    # <--- 主要修改在这里：动态计算sigmas --->
    try:
        # 简单的OLS回归来估计残差的尺度
        ols_beta = np.linalg.solve(X.T @ X + 1e-6 * np.eye(p), X.T @ y)  # 加一点正则项更稳定
        ols_residuals = y - X @ ols_beta
        std_resid = np.std(ols_residuals)

        # 基于基准设置不同状态的sigma
        sigmas = np.array([0.75 * std_resid, 1.25 * std_resid])

        # if verbose:
        #     print(
        #         f"根据数据动态初始化: 估计的残差标准差 = {std_resid:.4f}, 初始sigmas = [{sigmas[0]:.4f}, {sigmas[1]:.4f}]")

    except np.linalg.LinAlgError:
        # if verbose:
        #     print("OLS初始化失败，使用默认sigma值")
        sigmas = np.array([0.5, 0.8])
    # <--- 修改结束 --->

    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    return betas, sigmas, P, pi

def forward_backward(y, X, betas, sigmas, P, pi):
    """
    前向-后向算法实现，返回对数似然
    """
    T = len(y)
    K = len(sigmas)

    log_lik = np.zeros((T, K))
    for k in range(K):
        # <--- 修改在这里：增加对sigma过小的保护 --->
        safe_sigma = np.maximum(sigmas[k], 1e-8)
        residuals = y - X @ betas[k]
        log_lik[:, k] = norm.logpdf(residuals, scale=safe_sigma)

    # ... (算法主体不变) ...
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

    # <--- 修改在这里：直接返回对数似然 --->
    log_likelihood = logsumexp(log_alpha[-1])
    return gamma, xi, log_likelihood

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
        W = np.diag(gamma[:, k])
        XtWX = X.T @ W @ X
        XtWX_inv = inv(XtWX + 1e-8 * np.eye(p))
        residuals = y - X @ betas[k]
        weighted_residuals_sq = gamma[:, k] * (residuals ** 2)
        weighted_sum = np.sum(gamma[:, k])
        mse = np.sum(weighted_residuals_sq) / (weighted_sum - p)
        beta_se[k] = np.sqrt(np.diag(XtWX_inv) * mse)
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
            P_se[i, j] = np.sqrt(p_ij * (1 - p_ij) / (state_count + 1e-10))

    return P_se


# em_mshar 函数的修改
# 修改 em_mshar
def em_mshar(data_rv, max_iter=100, tol=1e-6, verbose=True):
    """
    使用EM算法估计马尔可夫切换HAR-CJ模型的参数
    """
    X, y = prepare_X_y(data_rv)
    T, p = X.shape
    K = 2

    # <--- 修改初始化调用 --->
    betas, sigmas, P, pi = initialize_parameters(K, p, X, y, verbose=verbose)

    # <--- 修改为使用 log_likelihood --->
    old_log_likelihood = -np.inf
    converged = False
    iteration = 0

    for iteration in range(max_iter):
        # <--- 接收 log_likelihood --->
        gamma, xi, log_likelihood = forward_backward(y, X, betas, sigmas, P, pi)

        # M步更新... (逻辑不变，但加一点保护)
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
                if sigmas[k] < 1e-8:  # 保护措施
                    sigmas[k] = 1e-8
            else:
                sigmas[k] = 0.01

        P = xi.sum(axis=0)
        row_sums = P.sum(axis=1, keepdims=True)
        P = (P + 1e-10) / (row_sums + K * 1e-10)
        pi = gamma[0]

        # <--- 修改收敛检查 --->
        if not np.isfinite(log_likelihood):
            if verbose: print(f"迭代 {iteration}: 对数似然值变为非有限数，停止。")
            break

        if abs(log_likelihood - old_log_likelihood) < tol and iteration > 0:
            converged = True
            break

        old_log_likelihood = log_likelihood

    # ... 计算标准误和t统计量 (代码不变) ...
    beta_se, sigma_se = calculate_standard_errors(X, y, betas, sigmas, gamma, K, p)
    P_se = calculate_transition_prob_se(xi, gamma, T)
    t_stats = betas / (beta_se + 1e-10)  # 避免除以0
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

    # <--- 修改模型信息收集 --->
    num_params = K * p + K + K * (K - 1)  # 总参数个数：K*p个beta + K个sigma + K*(K-1)个转移概率
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
        'log_likelihood': log_likelihood,  # 直接存储 log_likelihood
        'iterations': iteration + 1,
        'converged': converged,
        'AIC': -2 * log_likelihood + 2 * num_params,
        'BIC': -2 * log_likelihood + np.log(T) * num_params
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
        data_rv['Jv_lag1'].values,
        data_rv['Jv_lag5'].values,
        data_rv['Jv_lag22'].values,
        data_rv['C_t_lag1'].values,
        data_rv['C_t_lag5'].values,
        data_rv['C_t_lag22'].values
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
        data_subset['Jv_lag1'].values,
        data_subset['Jv_lag5'].values,
        data_subset['Jv_lag22'].values,
        data_subset['C_t_lag1'].values,
        data_subset['C_t_lag5'].values,
        data_subset['C_t_lag22'].values
    ])
    return X

def mshar_predict_1step(X, betas, sigmas, P, gamma_last):
    """
    使用马尔可夫切换HAR-CJ模型进行一步预测

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
    regime_predictions = np.array([X @ betas[k] for k in range(K)])
    prediction = np.sum(regime_predictions * gamma_last)
    return prediction
def rolling_window_mshar_prediction(train_data, test_data, horizons=[1, 5, 22], window_size=None):
    """
    使用滚动窗口进行马尔可夫切换HAR-RV模型预测，去掉指数还原

    参数:
    train_data: 训练数据（包含 Date, RV 等列）
    test_data: 测试数据（包含 Date, RV 等列）
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
                print("\n===== 第一步模型参数 =====")
                for k in range(2):
                    print(f"状态{k + 1}系数:")
                    param_names = ['常数项', 'Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']
                    for j in range(len(param_names)):
                        print(
                            f"  {param_names[j]}: {betas[k][j]:.6f} (标准误: {model_info['beta_se'][k][j]:.6f}, "
                            f"t值: {model_info['t_stats'][k][j]:.4f}, p值: {model_info['p_values'][k][j]:.4f})")
                    print(f"状态{k + 1}波动率: {sigmas[k]:.6f} (标准误: {model_info['sigma_se'][k]:.6f})")
                print(f"转移概率矩阵:")
                print(f"  P00: {P[0][0]:.4f} (标准误: {model_info['P_se'][0][0]:.4f})")
                print(f"  P11: {P[1][1]:.4f} (标准误: {model_info['P_se'][1][1]:.4f})")
                print(f"初始状态概率: [{pi[0]:.4f}, {pi[1]:.4f}]")
                print(f"对数似然值: {model_info['log_likelihood']:.4f}")
                print(f"AIC: {model_info['AIC']:.4f}")
                print(f"BIC: {model_info['BIC']:.4f}")
                print("=========================\n")


            # 一步预测
            X_1step = prepare_step_features(test_data.iloc[i:i + 1])
            pred_1 = mshar_predict_1step(X_1step[0], betas, sigmas, P, gamma[-1])
            predictions_dict[1].append(pred_1)
            actuals_dict[1].append(test_data['RV'].iloc[i])

            # 五步预测
            if i + 4 < n_test_points:
                X_5step = prepare_step_features(test_data.iloc[i + 4:i + 5])
                pred_5 = mshar_predict_1step(X_5step[0], betas, sigmas, P, gamma[-1])
                predictions_dict[5].append(pred_5)
                actuals_dict[5].append(test_data['RV'].iloc[i + 4])
            else:
                predictions_dict[5].append(None)
                actuals_dict[5].append(None)

            # 22步预测
            if i + 21 < n_test_points:
                X_22step = prepare_step_features(test_data.iloc[i + 21:i + 22])
                pred_22 = mshar_predict_1step(X_22step[0], betas, sigmas, P, gamma[-1])
                predictions_dict[22].append(pred_22)
                actuals_dict[22].append(test_data['RV'].iloc[i + 21])
            else:
                predictions_dict[22].append(None)
                actuals_dict[22].append(None)

        except Exception as e:
            print(f"预测点 {i} 发生错误: {str(e)}")
            for h in horizons:
                predictions_dict[h].append(None)
                if i + h - 1 < n_test_points:
                    actuals_dict[h].append(test_data['RV'].iloc[i + h - 1])
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
                       output_dir='mshar_cj_results'):
    """
    运行马尔可夫切换HAR-CJ模型的预测

    参数:
    train_data: 训练数据
    test_data: 测试数据
    horizons: 预测步长列表
    window_size: 滚动窗口大小
    output_dir: 结果保存目录

    返回:
    results_dict: 包含预测结果和实际值的字典
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("开始滚动窗口预测...")
    predictions_dict, actuals_dict, first_step_params = rolling_window_mshar_prediction(
        train_data, test_data, horizons, window_size
    )
    print("预测完成")

    results_dict = {'predictions': {}, 'actuals': {}, 'first_step_params': first_step_params}

    for h in horizons:
        results_df = pd.DataFrame({
            f'预测_RV_{h}步': predictions_dict[h],
            f'实际_RV_{h}步': actuals_dict[h]
        })
        if hasattr(test_data, 'index') and len(test_data.index) >= len(results_df):
            results_df.index = test_data.index[:len(results_df)]
        h_save_path = os.path.join(output_dir, f'mshar_cj_forecast_h{h}.csv')
        results_df.to_csv(h_save_path)
        print(f"已保存 {h} 步预测结果")

        results_dict['predictions'][h] = predictions_dict[h]
        results_dict['actuals'][h] = actuals_dict[h]

    all_results = pd.DataFrame()
    for h in horizons:
        all_results[f'预测_RV_{h}步'] = predictions_dict[h]
        all_results[f'实际_RV_{h}步'] = actuals_dict[h]
    if hasattr(test_data, 'index') and len(test_data.index) >= len(all_results):
        all_results.index = test_data.index[:len(all_results)]
    all_save_path = os.path.join(output_dir, 'mshar_cj_forecast_all_horizons.csv')
    all_results.to_csv(all_save_path)
    print(f"已保存所有预测步长的结果")

    param_names = ['常数项', 'Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']
    parameter_info = {
        '模型信息': {
            '对数似然值': float(np.log(first_step_params['likelihood'])),
            'AIC': float(first_step_params['AIC']),
            'BIC': float(first_step_params['BIC']),
            '迭代次数': int(first_step_params['iterations']),
            '是否收敛': bool(first_step_params['converged'])
        }
    }
    for k in range(2):
        state_info = {}
        for j, name in enumerate(param_names):
            state_info[f'{name}_系数'] = float(first_step_params['betas'][k][j])
            state_info[f'{name}_标准误'] = float(first_step_params['beta_se'][k][j])
            state_info[f'{name}_t值'] = float(first_step_params['t_stats'][k][j])
            state_info[f'{name}_p值'] = float(first_step_params['p_values'][k][j])
        state_info['波动率'] = float(first_step_params['sigmas'][k])
        state_info['波动率_标准误'] = float(first_step_params['sigma_se'][k])
        parameter_info[f'状态{k + 1}'] = state_info

    parameter_info['转移概率'] = {
        'P00': float(first_step_params['P'][0][0]),
        'P00_标准误': float(first_step_params['P_se'][0][0]),
        'P11': float(first_step_params['P'][1][1]),
        'P11_标准误': float(first_step_params['P_se'][1][1]),
    }

    params_path = os.path.join(output_dir, 'mshar_cj_first_step_parameters.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(parameter_info, f, indent=4, ensure_ascii=False)
    print(f"已保存第一步模型参数到: {params_path}")

    param_rows = []
    for key, value in parameter_info['模型信息'].items():
        param_rows.append({'参数': key, '值': value})
    for k in range(2):
        state_key = f'状态{k + 1}'
        for param, value in parameter_info[state_key].items():
            param_rows.append({'参数': f'{state_key}_{param}', '值': value})
    for param, value in parameter_info['转移概率'].items():
        param_rows.append({'参数': param, '值': value})
    param_df = pd.DataFrame(param_rows)
    param_csv_path = os.path.join(output_dir, 'mshar_cj_first_step_parameters.csv')
    param_df.to_csv(param_csv_path, index=False, encoding='utf-8')
    print(f"已保存第一步模型参数到CSV: {param_csv_path}")

    return results_dict

def run_model_example():
    """
    运行模型示例
    注意: 需要提前准备好train_data和test_data
    train_data和test_data必须包含'RV', 'Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22'列
    """
    output_dir = 'mshar_cj_results'
    results = run_mshar_forecast(
        train_data=train_data,
        test_data=test_data,
        horizons=[1, 5, 22],
        window_size=None,
        output_dir=output_dir
    )
    return results

if __name__ == "__main__":
    run_model_example()
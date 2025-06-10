import pandas as pd
import numpy as np
from scipy.stats import norm
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

# 定义数据文件路径
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/dash.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/eth.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ltc.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/xlm.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/xrp.csv"
}


def get_re(data, alpha):
    """
    计算REX指标的函数
    """
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
        group.loc[group.index[0], 'Ret'] = 0  # First return is 0
        group.loc[group['close'].shift(1) == 0, 'Ret'] = np.nan  # Handle division by zero

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
    result = result.groupby('day', group_keys=False).apply(calculate_daily_metrics, include_groups=False).reset_index()
    # 过滤掉None值
    result = result.dropna()

    return result


# 存储所有加密货币的REX数据
all_rex_data = {}

# 处理每个加密货币文件
for crypto_name, file_path in data_files.items():
    print(f"正在处理 {crypto_name}...")

    try:
        # 读取数据
        df = pd.read_csv(file_path)

        # 如果文件中有code列，过滤对应的数据；否则直接使用所有数据
        if 'code' in df.columns:
            data_filtered = df[df['code'] == crypto_name].copy()
        else:
            data_filtered = df.copy()

        # 计算REX指标
        har_re = get_re(data_filtered, alpha=0.05)

        # 添加加密货币标识列
        har_re['crypto'] = crypto_name

        # 存储结果
        all_rex_data[crypto_name] = har_re

        print(f"{crypto_name} 处理完成，共 {len(har_re)} 天数据")

    except Exception as e:
        print(f"处理 {crypto_name} 时出错: {e}")

# 合并所有数据
if all_rex_data:
    # 将所有数据合并成一个DataFrame
    combined_data = pd.concat(all_rex_data.values(), ignore_index=True)

    # 创建三个分别的数据集
    # 1. REX_minus 数据 (包含day, crypto, REX_minus)
    rex_minus_data = combined_data[['day', 'crypto', 'REX_minus']].copy()
    rex_minus_pivot = rex_minus_data.pivot(index='day', columns='crypto', values='REX_minus')
    rex_minus_pivot.index.name = 'DT'
    all_RD = rex_minus_pivot

    # 2. REX_plus 数据 (包含day, crypto, REX_plus)
    rex_plus_data = combined_data[['day', 'crypto', 'REX_plus']].copy()
    rex_plus_pivot = rex_plus_data.pivot(index='day', columns='crypto', values='REX_plus')
    rex_plus_pivot.index.name = 'DT'
    all_RP = rex_plus_pivot


    # 3. REX_moderate 数据 (包含day, crypto, REX_moderate)
    rex_moderate_data = combined_data[['day', 'crypto', 'REX_moderate']].copy()
    rex_moderate_pivot = rex_moderate_data.pivot(index='day', columns='crypto', values='REX_moderate')
    rex_moderate_pivot.index.name = 'DT'
    all_RM = rex_moderate_pivot

print(all_RM)
print(all_RD)
print(all_RP)

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR



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
RV['DT'] = pd.to_datetime(RV['DT'])
RV = RV.set_index('DT')['RV']  # 将 'DT' 设为索引，提取 'RV' 列

re_m_lag1 = all_RM['BTC'].shift(1)
re_m_lag5 = all_RM['BTC'].rolling(window=5).mean().shift(1)
re_m_lag22 = all_RM['BTC'].rolling(window=22).mean().shift(1)
re_p_lag1 = all_RP['BTC'].shift(1)
re_p_lag5 = all_RP['BTC'].rolling(window=5).mean().shift(1)
re_p_lag22 = all_RP['BTC'].rolling(window=22).mean().shift(1)
re_d_lag1 = all_RD['BTC'].shift(1)
re_d_lag5 = all_RD['BTC'].rolling(window=5).mean().shift(1)
re_d_lag22 = all_RD['BTC'].rolling(window=22).mean().shift(1)

model1 = pd.DataFrame({
    'RV': RV,
    're_m_lag1': re_m_lag1,
    're_m_lag5': re_m_lag5,
    're_m_lag22': re_m_lag22,
    're_p_lag1': re_p_lag1,
    're_p_lag5': re_p_lag5,
    're_p_lag22': re_p_lag22,
    're_d_lag1': re_d_lag1,
    're_d_lag5': re_d_lag5,
    're_d_lag22': re_d_lag22,
    'BTC_lag1':all_RP['ETH'].shift(1)
}).dropna()

'''
test_size = 300

train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

X_train = train_data[['re_m_lag1', 're_p_lag1', 're_m_lag5', 're_p_lag5', 're_m_lag22', 're_p_lag22']]
y_train = train_data['RV']
X_test = test_data[['re_m_lag1', 're_p_lag1', 're_m_lag5', 're_p_lag5', 're_m_lag22', 're_p_lag22']]
y_test = test_data['RV']
z_train = train_data['BTC_lag1']
z_test = test_data['BTC_lag1']

rolling_X = X_train.values
rolling_y = y_train.values
rolling_z = z_train.values

predictions_tvtp1 = []
actuals_tvtp1 = []
predictions_tvtp5 = []
actuals_tvtp5 = []
predictions_tvtp22 = []
actuals_tvtp22 = []

k_features = X_train.shape[1]
k = k_features + 1
n_states = 2
n_params = k * n_states + n_states + 2 * n_states
initial_params = np.zeros(n_params)

ols_model = sm.OLS(rolling_y, np.column_stack([np.ones(len(rolling_y)), rolling_X])).fit()
for s in range(n_states):
    factor = 0.6 + 0.8 * s
    initial_params[s * k:(s + 1) * k] = ols_model.params * factor + np.random.normal(0, 0.05, k)

residuals = rolling_y - np.column_stack([np.ones(len(rolling_y)), rolling_X]) @ ols_model.params
sigma_base = np.std(residuals)
for s in range(n_states):
    initial_params[k * n_states + s] = np.log(sigma_base * (0.5 + s) + 1e-6)  # Add epsilon for stability

initial_params[k * n_states + n_states:k * n_states + 2 * n_states] = [0.8, 0.8]
initial_params[k * n_states + 2 * n_states:] = [-0.1, 0.1]

# Initialize last_filtered_probs_normalized before the loop for the first iteration's potential needs
last_filtered_probs_normalized = np.ones(n_states) / n_states


def tvtp_ms_har_log_likelihood(params, y, X_with_const, z, n_states=2, return_filtered_probs=False):
    n = len(y)
    k_with_const = X_with_const.shape[1]

    beta = params[:k_with_const * n_states].reshape(n_states, k_with_const)
    sigma = np.exp(params[k_with_const * n_states:k_with_const * n_states + n_states])
    a = params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
    b = params[k_with_const * n_states + 2 * n_states:]

    log_filtered_prob = np.zeros((n, n_states))
    log_lik = 0.0

    mu_cache = np.zeros((n, n_states))
    for j in range(n_states):
        mu_cache[:, j] = X_with_const @ beta[j]

    for t in range(n):
        P = np.zeros((n_states, n_states))
        if t > 0:
            logit_11 = np.clip(a[0] + b[0] * z[t - 1], -30, 30)
            p11 = 1.0 / (1.0 + np.exp(-logit_11))
            logit_22 = np.clip(a[1] + b[1] * z[t - 1], -30, 30)
            p22 = 1.0 / (1.0 + np.exp(-logit_22))

            p11 = np.clip(p11, 0.001, 0.999)
            p22 = np.clip(p22, 0.001, 0.999)

            P[0, 0] = p11
            P[0, 1] = 1.0 - p11
            P[1, 0] = 1.0 - p22
            P[1, 1] = p22

            filtered_prob_prev = np.exp(log_filtered_prob[t - 1] - np.max(log_filtered_prob[t - 1]))  # Stability
            filtered_prob_prev = filtered_prob_prev / np.sum(filtered_prob_prev)
            pred_prob = filtered_prob_prev @ P
        else:
            pred_prob = np.ones(n_states) / n_states

        pred_prob = np.clip(pred_prob, 1e-10, 1.0)
        pred_prob /= np.sum(pred_prob)

        log_conditional_densities = np.zeros(n_states)
        for j in range(n_states):
            log_conditional_densities[j] = norm.logpdf(y[t], mu_cache[t, j], sigma[j] + 1e-8)  # Add epsilon to sigma

        log_joint_prob = np.log(pred_prob) + log_conditional_densities
        max_log_prob = np.max(log_joint_prob)
        log_marginal_prob = max_log_prob + np.log(np.sum(np.exp(log_joint_prob - max_log_prob)))

        log_filtered_prob[t] = log_joint_prob - log_marginal_prob
        log_lik += log_marginal_prob

    if np.isnan(log_lik) or np.isinf(log_lik):
        log_lik = -1e10  # Return a very bad likelihood instead of erroring out optimizer

    if return_filtered_probs:
        return -log_lik, log_filtered_prob
    return -log_lik


def fit_tvtp_ms_har(y, X_with_const, z, n_states=2, initial_params=None):
    k_with_const = X_with_const.shape[1]
    # n_params defined globally
    if initial_params is None:  # Should ideally not happen if global initial_params is well-managed
        initial_params_fit = np.zeros(n_params)
        ols_model_fit = sm.OLS(y, X_with_const).fit()
        for s_fit in range(n_states):
            factor_fit = 0.6 + 0.8 * s_fit
            initial_params_fit[
            s_fit * k_with_const:(s_fit + 1) * k_with_const] = ols_model_fit.params * factor_fit + np.random.normal(0,
                                                                                                                    0.05,
                                                                                                                    k_with_const)
        residuals_fit = y - X_with_const @ ols_model_fit.params
        sigma_base_fit = np.std(residuals_fit)
        for s_fit in range(n_states):
            initial_params_fit[k_with_const * n_states + s_fit] = np.log(sigma_base_fit * (0.5 + s_fit) + 1e-6)
        initial_params_fit[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states] = [0.8, 0.8]
        initial_params_fit[k_with_const * n_states + 2 * n_states:] = [-0.1, 0.1]
    else:
        initial_params_fit = initial_params

    bounds = []
    for _ in range(k_with_const * n_states):
        bounds.append((None, None))
    for _ in range(n_states):
        bounds.append((np.log(0.0001), np.log(5 * np.std(y) + 1e-6)))
    for _ in range(n_states):  # a params
        bounds.append((-10, 10))  # Looser bounds for a

    z_std = np.std(z) if len(z) > 1 else 1.0  # Handle z being single element
    for _ in range(n_states):  # b params
        bound_scale = min(10 / max(0.1, z_std), 50)  # Looser bounds for b
        bounds.append((-bound_scale, bound_scale))

    result = minimize(
        tvtp_ms_har_log_likelihood,
        initial_params_fit,
        args=(y, X_with_const, z, n_states, False),  # Pass False for return_filtered_probs
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1500, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-6}  # Increased maxiter
    )

    final_log_filtered_probs = None
    if result.success:
        _, final_log_filtered_probs = tvtp_ms_har_log_likelihood(
            result.x, y, X_with_const, z, n_states, return_filtered_probs=True
        )
    # else:
    # print(f"Optimization failed in fit_tvtp_ms_har: {result.message}")

    return result, final_log_filtered_probs


def predict_tvtp_corrected(X_pred_features, z_for_P_matrix,
                           last_filt_probs_norm,  # P(S_T | data_1..T)
                           beta_est,  # n_states x k_with_const
                           a_est, b_est, n_states_pred=2):
    X_pred_with_const = np.column_stack([np.ones(len(X_pred_features)), X_pred_features])
    mu = np.dot(X_pred_with_const, beta_est.T)  # Shape: (num_preds, n_states)

    P_matrix = np.zeros((n_states_pred, n_states_pred))

    # Ensure z_for_P_matrix is a scalar if X_pred_features is for a single step
    # If z_for_P_matrix is an array (e.g. if passed from z_train_vals[-1]), ensure it's scalar
    if isinstance(z_for_P_matrix, (np.ndarray, pd.Series)):
        z_val_for_P = z_for_P_matrix.item() if z_for_P_matrix.size == 1 else z_for_P_matrix[
            0]  # Or some other logic if z_for_P_matrix can be multi-element
    else:
        z_val_for_P = z_for_P_matrix

    logit_11 = np.clip(a_est[0] + b_est[0] * z_val_for_P, -30, 30)
    p11 = 1.0 / (1.0 + np.exp(-logit_11))
    p11 = np.clip(p11, 0.001, 0.999)

    logit_22 = np.clip(a_est[1] + b_est[1] * z_val_for_P, -30, 30)
    p22 = 1.0 / (1.0 + np.exp(-logit_22))
    p22 = np.clip(p22, 0.001, 0.999)

    P_matrix[0, 0] = p11
    P_matrix[0, 1] = 1.0 - p11
    P_matrix[1, 0] = 1.0 - p22
    P_matrix[1, 1] = p22

    # Predicted state probabilities for the next step: P(S_{T+1}|data_T, z_T) = P(S_T|data_T) @ P_matrix(z_T)
    pred_state_probs = last_filt_probs_norm @ P_matrix
    pred_state_probs = np.clip(pred_state_probs, 1e-10, 1.0)
    pred_state_probs /= np.sum(pred_state_probs)

    pred = np.sum(pred_state_probs * mu[0])  # mu[0] assumes X_pred_features is for one time step
    return pred


print("开始滚动窗口预测...")
for i in tqdm(range(len(X_test))):
    X_train_vals = rolling_X
    y_train_vals = rolling_y
    z_train_vals = rolling_z

    X_train_with_const = np.column_stack([np.ones(len(X_train_vals)), X_train_vals])

    try:
        tvtp_fit_result, final_log_filtered_probs_train = fit_tvtp_ms_har(
            y_train_vals, X_train_with_const, z_train_vals, n_states, initial_params=initial_params
        )
        if tvtp_fit_result.success:
            print(f"时间窗 {i}: 模型拟合成功。")
            fit_successful_this_iteration = True
        else:
            print(f"时间窗 {i}: 模型拟合失败。原因: {tvtp_fit_result.message}")
            fit_successful_this_iteration = False  # Explicitly set
        if tvtp_fit_result.success and final_log_filtered_probs_train is not None:
            initial_params = tvtp_fit_result.x
            # Get the filtered probabilities from the last observation of the training window
            last_log_probs = final_log_filtered_probs_train[-1, :]
            # Normalize carefully
            exp_last_log_probs = np.exp(last_log_probs - np.max(last_log_probs))
            last_filtered_probs_normalized = exp_last_log_probs / np.sum(exp_last_log_probs)

        # else: if fit fails, initial_params and last_filtered_probs_normalized from previous iter are used

        k_with_const = X_train_with_const.shape[1]
        beta_tvtp = initial_params[:k_with_const * n_states].reshape(n_states, k_with_const)
        # sigma_tvtp = np.exp(initial_params[k_with_const * n_states:k_with_const * n_states + n_states]) # Not directly used in point prediction function shown
        a_params = initial_params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
        b_params = initial_params[k_with_const * n_states + 2 * n_states:]

        # 1步预测
        current_X_test_features = X_test.iloc[i:i + 1].values
        # z_for_P_matrix is the z value that influences the transition INTO the current_X_test time step
        z_for_P_matrix = z_test.iloc[i - 1] if i > 0 else (z_train_vals[-1] if len(z_train_vals) > 0 else 0.0)

        pred_1 = predict_tvtp_corrected(
            current_X_test_features,
            z_for_P_matrix,
            last_filtered_probs_normalized,  # P(S_T | training data up to T)
            beta_tvtp,
            a_params,
            b_params,
            n_states
        )
        predictions_tvtp1.append(float(pred_1))
        actuals_tvtp1.append(float(y_test.iloc[i]))
        if i % 50 == 0 or i == len(X_test) - 1:  # Print less frequently
            print(f"时间窗 {i} - 1步预测值: {pred_1:.4f}, 实际值: {y_test.iloc[i]:.4f}")

        # 5步预测
        if i + 4 < len(X_test):
            future_X_test_5_features = X_test.iloc[i + 4:i + 5].values
            # z_for_P_matrix_5 is z that influences transition into X_test.iloc[i+4]
            z_for_P_matrix_5 = z_test.iloc[i + 3] if i + 3 < len(z_test) else (
                z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_5 = predict_tvtp_corrected(
                future_X_test_5_features,
                z_for_P_matrix_5,
                last_filtered_probs_normalized,  # Still using P(S_T | training data up to T)
                beta_tvtp,
                a_params,
                b_params,
                n_states
            )
            predictions_tvtp5.append(float(pred_5))
            actuals_tvtp5.append(float(y_test.iloc[i + 4]))
        else:
            predictions_tvtp5.append(None)
            actuals_tvtp5.append(None)

        # 22步预测
        if i + 21 < len(X_test):
            future_X_test_22_features = X_test.iloc[i + 21:i + 22].values
            z_for_P_matrix_22 = z_test.iloc[i + 20] if i + 20 < len(z_test) else (
                z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_22 = predict_tvtp_corrected(
                future_X_test_22_features,
                z_for_P_matrix_22,
                last_filtered_probs_normalized,  # Still using P(S_T | training data up to T)
                beta_tvtp,
                a_params,
                b_params,
                n_states
            )
            predictions_tvtp22.append(float(pred_22))
            actuals_tvtp22.append(float(y_test.iloc[i + 21]))
        else:
            predictions_tvtp22.append(None)
            actuals_tvtp22.append(None)

    except Exception as e:
        print(f"迭代 {i} 出现错误: {str(e)}")
        predictions_tvtp1.append(predictions_tvtp1[-1] if predictions_tvtp1 else np.nan)
        actuals_tvtp1.append(float(y_test.iloc[i]))

        predictions_tvtp5.append(
            predictions_tvtp5[-1] if predictions_tvtp5 and predictions_tvtp5[-1] is not None else np.nan)
        actuals_tvtp5.append(float(y_test.iloc[i + 4]) if i + 4 < len(y_test) else np.nan)

        predictions_tvtp22.append(
            predictions_tvtp22[-1] if predictions_tvtp22 and predictions_tvtp22[-1] is not None else np.nan)
        actuals_tvtp22.append(float(y_test.iloc[i + 21]) if i + 21 < len(y_test) else np.nan)

    new_X = X_test.iloc[i:i + 1].values
    rolling_X = np.vstack((rolling_X[1:], new_X))
    rolling_y = np.append(rolling_y[1:], y_test.iloc[i])
    rolling_z = np.append(rolling_z[1:], z_test.iloc[i])

print("预测完成!")

df_predictions_tvtp = pd.DataFrame({
    'Prediction_1': predictions_tvtp1,
    'Actual_1': actuals_tvtp1,
    'Prediction_5': predictions_tvtp5,
    'Actual_5': actuals_tvtp5,
    'Prediction_22': predictions_tvtp22,
    'Actual_22': actuals_tvtp22
})

df_predictions_tvtp.to_csv('tvtphar-re_corrected_DASH.csv', index=False)
print("结果已保存到 'tvtphar-re_corrected_DASH.csv'")
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm

# Define window and test sizes
window_size = 1000  # Rolling window size
test_size = 500     # Test set size

# Split training and test data
train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

X_train = train_data[['re_m_lag1', 're_p_lag1', 're_m_lag5', 're_p_lag5', 're_m_lag22', 're_p_lag22']]
y_train = train_data['RV']
X_test = test_data[['re_m_lag1', 're_p_lag1', 're_m_lag5', 're_p_lag5', 're_m_lag22', 're_p_lag22']]
y_test = test_data['RV']
z_train = train_data['BTC_lag1']
z_test = test_data['BTC_lag1']

rolling_X = X_train.values
rolling_y = y_train.values
rolling_z = z_train.values

# Initialize prediction and actual lists
predictions_tvtp1 = []
actuals_tvtp1 = []
predictions_tvtp5 = []
actuals_tvtp5 = []
predictions_tvtp22 = []
actuals_tvtp22 = []

# Initialize model parameters
k_features = X_train.shape[1]
k = k_features + 1
n_states = 2
n_params = k * n_states + n_states + 2 * n_states
initial_params = np.zeros(n_params)

# OLS initialization
ols_model = sm.OLS(rolling_y, np.column_stack([np.ones(len(rolling_y)), rolling_X])).fit()
for s in range(n_states):
    factor = 0.6 + 0.8 * s
    initial_params[s * k:(s + 1) * k] = ols_model.params * factor + np.random.normal(0, 0.05, k)

residuals = rolling_y - np.column_stack([np.ones(len(rolling_y)), rolling_X]) @ ols_model.params
sigma_base = np.std(residuals)
for s in range(n_states):
    initial_params[k * n_states + s] = np.log(sigma_base * (0.5 + s) + 1e-6)

initial_params[k * n_states + n_states:k * n_states + 2 * n_states] = [0.8, 0.8]
initial_params[k * n_states + 2 * n_states:] = [-0.1, 0.1]

# Initialize filtered probabilities
last_filtered_probs_normalized = np.ones(n_states) / n_states

def tvtp_ms_har_log_likelihood(params, y, X_with_const, z, n_states=2, return_filtered_probs=False):
    n = len(y)
    k_with_const = X_with_const.shape[1]

    beta = params[:k_with_const * n_states].reshape(n_states, k_with_const)
    sigma = np.exp(params[k_with_const * n_states:k_with_const * n_states + n_states])
    a = params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
    b = params[k_with_const * n_states + 2 * n_states:]

    log_filtered_prob = np.zeros((n, n_states))
    log_lik = 0.0

    mu_cache = np.zeros((n, n_states))
    for j in range(n_states):
        mu_cache[:, j] = X_with_const @ beta[j]

    for t in range(n):
        P = np.zeros((n_states, n_states))
        if t > 0:
            logit_11 = np.clip(a[0] + b[0] * z[t - 1], -30, 30)
            p11 = 1.0 / (1.0 + np.exp(-logit_11))
            logit_22 = np.clip(a[1] + b[1] * z[t - 1], -30, 30)
            p22 = 1.0 / (1.0 + np.exp(-logit_22))

            p11 = np.clip(p11, 0.001, 0.999)
            p22 = np.clip(p22, 0.001, 0.999)

            P[0, 0] = p11
            P[0, 1] = 1.0 - p11
            P[1, 0] = 1.0 - p22
            P[1, 1] = p22

            filtered_prob_prev = np.exp(log_filtered_prob[t - 1] - np.max(log_filtered_prob[t - 1]))
            filtered_prob_prev = filtered_prob_prev / np.sum(filtered_prob_prev)
            pred_prob = filtered_prob_prev @ P
        else:
            pred_prob = np.ones(n_states) / n_states

        log_conditional_densities = np.zeros(n_states)
        for j in range(n_states):
            log_conditional_densities[j] = norm.logpdf(y[t], mu_cache[t, j], sigma[j] + 1e-8)

        log_joint_prob = np.log(pred_prob) + log_conditional_densities
        max_log_prob = np.max(log_joint_prob)
        log_marginal_prob = max_log_prob + np.log(np.sum(np.exp(log_joint_prob - max_log_prob)))

        log_filtered_prob[t] = log_joint_prob - log_marginal_prob
        log_lik += log_marginal_prob

    if np.isnan(log_lik) or np.isinf(log_lik):
        log_lik = -1e10

    if return_filtered_probs:
        return -log_lik, log_filtered_prob
    return -log_lik

def fit_tvtp_ms_har(y, X_with_const, z, n_states=2, initial_params=None):
    k_with_const = X_with_const.shape[1]
    if initial_params is None:
        initial_params_fit = np.zeros(n_params)
        ols_model_fit = sm.OLS(y, X_with_const).fit()
        for s_fit in range(n_states):
            factor_fit = 0.6 + 0.8 * s_fit
            initial_params_fit[s_fit * k_with_const:(s_fit + 1) * k_with_const] = ols_model_fit.params * factor_fit + np.random.normal(0, 0.05, k_with_const)
        residuals_fit = y - X_with_const @ ols_model_fit.params
        sigma_base_fit = np.std(residuals_fit)
        for s_fit in range(n_states):
            initial_params_fit[k_with_const * n_states + s_fit] = np.log(sigma_base_fit * (0.5 + s_fit) + 1e-6)
        initial_params_fit[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states] = [0.8, 0.8]
        initial_params_fit[k_with_const * n_states + 2 * n_states:] = [-0.1, 0.1]
    else:
        initial_params_fit = initial_params

    bounds = []
    for _ in range(k_with_const * n_states):
        bounds.append((None, None))
    for _ in range(n_states):
        bounds.append((np.log(0.0001), np.log(5 * np.std(y) + 1e-6)))
    for _ in range(n_states):
        bounds.append((-10, 10))
    z_std = np.std(z) if len(z) > 1 else 1.0
    for _ in range(n_states):
        bound_scale = min(10 / max(0.1, z_std), 50)
        bounds.append((-bound_scale, bound_scale))

    result = minimize(
        tvtp_ms_har_log_likelihood,
        initial_params_fit,
        args=(y, X_with_const, z, n_states, False),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1500, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-6}
    )

    final_log_filtered_probs = None
    if result.success:
        _, final_log_filtered_probs = tvtp_ms_har_log_likelihood(
            result.x, y, X_with_const, z, n_states, return_filtered_probs=True
        )

    return result, final_log_filtered_probs

def predict_tvtp_corrected(X_pred_features, z_for_P_matrix, last_filt_probs_norm, beta_est, a_est, b_est, n_states_pred=2):
    X_pred_with_const = np.column_stack([np.ones(len(X_pred_features)), X_pred_features])
    mu = np.dot(X_pred_with_const, beta_est.T)

    P_matrix = np.zeros((n_states_pred, n_states_pred))
    if isinstance(z_for_P_matrix, (np.ndarray, pd.Series)):
        z_val_for_P = z_for_P_matrix.item() if z_for_P_matrix.size == 1 else z_for_P_matrix[0]
    else:
        z_val_for_P = z_for_P_matrix

    logit_11 = np.clip(a_est[0] + b_est[0] * z_val_for_P, -30, 30)
    p11 = 1.0 / (1.0 + np.exp(-logit_11))
    p11 = np.clip(p11, 0.001, 0.999)

    logit_22 = np.clip(a_est[1] + b_est[1] * z_val_for_P, -30, 30)
    p22 = 1.0 / (1.0 + np.exp(-logit_22))
    p22 = np.clip(p22, 0.001, 0.999)

    P_matrix[0, 0] = p11
    P_matrix[0, 1] = 1.0 - p11
    P_matrix[1, 0] = 1.0 - p22
    P_matrix[1, 1] = p22

    pred_state_probs = last_filt_probs_norm @ P_matrix
    pred_state_probs = np.clip(pred_state_probs, 1e-10, 1.0)
    pred_state_probs /= np.sum(pred_state_probs)

    pred = np.sum(pred_state_probs * mu[0])
    return pred

print("开始滚动窗口预测...")
for i in tqdm(range(len(X_test))):
    # Ensure rolling window size does not exceed window_size
    if len(rolling_X) > window_size:
        rolling_X = rolling_X[-window_size:]
        rolling_y = rolling_y[-window_size:]
        rolling_z = rolling_z[-window_size:]

    X_train_vals = rolling_X
    y_train_vals = rolling_y
    z_train_vals = rolling_z

    X_train_with_const = np.column_stack([np.ones(len(X_train_vals)), X_train_vals])

    try:
        tvtp_fit_result, final_log_filtered_probs_train = fit_tvtp_ms_har(
            y_train_vals, X_train_with_const, z_train_vals, n_states, initial_params=initial_params
        )
        if tvtp_fit_result.success:
            print(f"时间窗 {i}: 模型拟合成功。")
            initial_params = tvtp_fit_result.x
            last_log_probs = final_log_filtered_probs_train[-1, :]
            exp_last_log_probs = np.exp(last_log_probs - np.max(last_log_probs))
            last_filtered_probs_normalized = exp_last_log_probs / np.sum(exp_last_log_probs)
        else:
            print(f"时间窗 {i}: 模型拟合失败。原因: {tvtp_fit_result.message}")

        k_with_const = X_train_with_const.shape[1]
        beta_tvtp = initial_params[:k_with_const * n_states].reshape(n_states, k_with_const)
        a_params = initial_params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
        b_params = initial_params[k_with_const * n_states + 2 * n_states:]

        # 1-step prediction
        current_X_test_features = X_test.iloc[i:i + 1].values
        z_for_P_matrix = z_test.iloc[i - 1] if i > 0 else (z_train_vals[-1] if len(z_train_vals) > 0 else 0.0)
        pred_1 = predict_tvtp_corrected(
            current_X_test_features, z_for_P_matrix, last_filtered_probs_normalized, beta_tvtp, a_params, b_params, n_states
        )
        predictions_tvtp1.append(float(pred_1))
        actuals_tvtp1.append(float(y_test.iloc[i]))
        if i % 50 == 0 or i == len(X_test) - 1:
            print(f"时间窗 {i} - 1步预测值: {pred_1:.4f}, 实际值: {y_test.iloc[i]:.4f}")

        # 5-step prediction
        if i + 4 < len(X_test):
            future_X_test_5_features = X_test.iloc[i + 4:i + 5].values
            z_for_P_matrix_5 = z_test.iloc[i + 3] if i + 3 < len(z_test) else (z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_5 = predict_tvtp_corrected(
                future_X_test_5_features, z_for_P_matrix_5, last_filtered_probs_normalized, beta_tvtp, a_params, b_params, n_states
            )
            predictions_tvtp5.append(float(pred_5))
            actuals_tvtp5.append(float(y_test.iloc[i + 4]))
        else:
            predictions_tvtp5.append(None)
            actuals_tvtp5.append(None)

        # 22-step prediction
        if i + 21 < len(X_test):
            future_X_test_22_features = X_test.iloc[i + 21:i + 22].values
            z_for_P_matrix_22 = z_test.iloc[i + 20] if i + 20 < len(z_test) else (z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_22 = predict_tvtp_corrected(
                future_X_test_22_features, z_for_P_matrix_22, last_filtered_probs_normalized, beta_tvtp, a_params, b_params, n_states
            )
            predictions_tvtp22.append(float(pred_22))
            actuals_tvtp22.append(float(y_test.iloc[i + 21]))
        else:
            predictions_tvtp22.append(None)
            actuals_tvtp22.append(None)

        # Update rolling window
        new_X = X_test.iloc[i:i + 1].values
        rolling_X = np.vstack((rolling_X, new_X))
        rolling_y = np.append(rolling_y, y_test.iloc[i])
        rolling_z = np.append(rolling_z, z_test.iloc[i])

    except Exception as e:
        print(f"迭代 {i} 出现错误: {str(e)}")
        predictions_tvtp1.append(predictions_tvtp1[-1] if predictions_tvtp1 else np.nan)
        actuals_tvtp1.append(float(y_test.iloc[i]))
        predictions_tvtp5.append(predictions_tvtp5[-1] if predictions_tvtp5 and predictions_tvtp5[-1] is not None else np.nan)
        actuals_tvtp5.append(float(y_test.iloc[i + 4]) if i + 4 < len(y_test) else np.nan)
        predictions_tvtp22.append(predictions_tvtp22[-1] if predictions_tvtp22 and predictions_tvtp22[-1] is not None else np.nan)
        actuals_tvtp22.append(float(y_test.iloc[i + 21]) if i + 21 < len(y_test) else np.nan)

        # Update rolling window even in case of error
        new_X = X_test.iloc[i:i + 1].values
        rolling_X = np.vstack((rolling_X, new_X))
        rolling_y = np.append(rolling_y, y_test.iloc[i])
        rolling_z = np.append(rolling_z, z_test.iloc[i])

print("预测完成!")

# Save results
df_predictions_tvtp = pd.DataFrame({
    'Prediction_1': predictions_tvtp1,
    'Actual_1': actuals_tvtp1,
    'Prediction_5': predictions_tvtp5,
    'Actual_5': actuals_tvtp5,
    'Prediction_22': predictions_tvtp22,
    'Actual_22': actuals_tvtp22
})

df_predictions_tvtp.to_csv('tvtphar-re_window.csv', index=False)
print("结果已保存到 'tvtphar-re_window.csv'")
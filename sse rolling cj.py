
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

from copulae import TCopula
# Read the data
# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/result.csv")

# List of market codes and their new names
market_codes = ["000001.XSHG", "SH.000016", "SH.000300", "SZ.399001", "SZ.399006"]
market_names = ["SSE", "SSE50", "CSI", "SZCZ", "STAR"]


# Define the get_RV_BV function
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
        # Calculate RV
        RV = np.sum(group['Ret'] ** 2)
        # Calculate BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())
        # Calculate TQ
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
        # Calculate JV and C_t
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV
        results.append({
            'day': day,
            'BV': BV,
            'JV': JV,
            'C_t': C_t
        })
    result_df = pd.DataFrame(results)
    return result_df.set_index('day')[['JV', 'C_t']]


# Compute results for each market and prepare all_JV and all_CT
all_JV = None
all_CT = None

for code, name in zip(market_codes, market_names):
    # Filter data for the current market
    data_filtered = df[df['code'] == code].copy()

    # Compute har_cj for the current market
    har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)

    # Reset index to get 'day' as a column
    har_cj = har_cj.reset_index()

    # Debug: Print columns of har_cj
    print(f"Columns for {name} ({code}): {har_cj.columns.tolist()}")

    # Create separate DataFrames for JV and C_t with market name
    jv_df = har_cj[['day', 'JV']].rename(columns={'JV': name})
    ct_df = har_cj[['day', 'C_t']].rename(columns={'C_t': name})

    # Merge with all_JV and all_CT
    if all_JV is None:
        all_JV = jv_df
        all_CT = ct_df
    else:
        all_JV = all_JV.merge(jv_df, on='day', how='outer')
        all_CT = all_CT.merge(ct_df, on='day', how='outer')

    # Debug: Print current columns of all_JV and all_CT
    print(f"Current all_JV columns: {all_JV.columns.tolist()}")
    print(f"Current all_CT columns: {all_CT.columns.tolist()}")

# Optional: If you want to rename 'day' to 'DT' at the end
all_JV = all_JV.rename(columns={'day': 'DT'})
all_CT = all_CT.rename(columns={'day': 'DT'})

# Ensure column order is correct
expected_columns = ["DT"] + market_names
if len(all_JV.columns) != len(expected_columns):
    raise ValueError(
        f"all_JV has {len(all_JV.columns)} columns, expected {len(expected_columns)}: {all_JV.columns.tolist()}")
if len(all_CT.columns) != len(expected_columns):
    raise ValueError(
        f"all_CT has {len(all_CT.columns)} columns, expected {len(expected_columns)}: {all_CT.columns.tolist()}")

# Reorder columns if needed
all_JV.columns = expected_columns
all_CT.columns = expected_columns


# Read the data
# Read the data
df_data = pd.read_csv("c:/Users/lenovo/Desktop/spillover/result.csv")

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
data_filtered = data_ret[data_ret['id'] == "000001.XSHG"].copy()
data_filtered = data_filtered.drop('id', axis=1)

# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x ** 2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']

# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])

# 数据准备
all_CT.columns = ["DT", "SSE", "SSE50", "CSI", "SZCZ", "STAR"]
all_JV.columns = ["DT", "SSE", "SSE50", "CSI", "SZCZ", "STAR"]

features = all_CT.drop(columns=['DT'])
data = features.to_numpy()

columns = ["SSE", "SSE50", "CSI", "SZCZ", "STAR"]
data_df = pd.DataFrame(data, columns=columns)

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


# 计算每个市场对SSE的净溢出效应(NPDC)
def npdc_to_sse(gfevd_array, columns):
    T = gfevd_array.shape[2]
    n = len(columns)
    npdc_sse = np.zeros((T, n))

    for t in range(T):
        for j in range(n):
            npdc_sse[t, j] = (gfevd_array[0, j, t] - gfevd_array[j, 0, t]) * 100

    npdc_df = pd.DataFrame(npdc_sse, columns=[f"NPDC_{col}_SSE" for col in columns])

    other_cols = [f"NPDC_{col}_SSE" for col in columns if col != 'SSE']
    npdc_df['Total_NPDC_others_SSE'] = npdc_df[other_cols].sum(axis=1)

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


# 计算总溢出效应
total_spillover_series = total_spillover(gfevd_results)

# 计算各市场对SSE的净溢出效应
npdc_df = npdc_to_sse(gfevd_results, columns)
print(npdc_df)
npdc = npdc_df.iloc[:, 1:-1]
# 选择需要分析的列（排除 SSE 自身和总和列）
npdc_columns = ['NPDC_SSE50_SSE', 'NPDC_CSI_SSE', 'NPDC_SZCZ_SSE', 'NPDC_STAR_SSE']

# 1. 哪个市场对SSE溢出最多（正值最大）
# 在每个时间点，找到NPDC最大值的市场
max_spillover_to_sse = npdc_df[npdc_columns].apply(lambda x: x.idxmax() if x.max() > 0 else None, axis=1)
max_spillover_to_sse_value = npdc_df[npdc_columns].max(axis=1)

# 统计每个市场对SSE溢出最多的频率
max_spillover_to_sse_counts = max_spillover_to_sse.value_counts()


# 输出结果
print("=== 哪个市场对SSE溢出最多 ===")
print("各市场对SSE溢出最多的频率：")
print(max_spillover_to_sse_counts)
print("\n平均NPDC值（正值表示对SSE的净溢出）：")
print(npdc_df[npdc_columns].mean())


JV_lag1 = all_JV['SSE'].shift(1)
C_t_lag1 = all_CT['SSE'].shift(1)
JV_lag5 = all_JV['SSE'].rolling(window=5).mean().shift(1)
C_t_lag5 = all_CT['SSE'].rolling(window=5).mean().shift(1)
JV_lag22 = all_JV['SSE'].rolling(window=22).mean().shift(1)
C_t_lag22 = all_CT['SSE'].rolling(window=22).mean().shift(1)


# 数据准备 (从您新提供的数据结构)
model1 = pd.DataFrame({
    'RV': np.log(RV['RV']),
    'Jv_lag1': JV_lag1,
    'Jv_lag5': JV_lag5,
    'Jv_lag22': JV_lag22,
    'C_t_lag1': C_t_lag1,
    'C_t_lag5': C_t_lag5,
    'C_t_lag22': C_t_lag22,
    'ndpc': all_CT['STAR'].shift(1)
}).dropna()
print(model1)
test_size = 300

train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

# 修改特征变量，使用新的JV和C_t相关特征
X_train = train_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22',
                      'C_t_lag1', 'C_t_lag5', 'C_t_lag22']]
y_train = train_data['RV']
X_test = test_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22',
                    'C_t_lag1', 'C_t_lag5', 'C_t_lag22']]
y_test = test_data['RV']
z_train = train_data['ndpc']  # 现在使用ndpc(STAR)作为转换概率的影响因子
z_test = test_data['ndpc']

rolling_X = X_train.values
rolling_y = y_train.values
rolling_z = z_train.values

predictions_tvtp1 = []
actuals_tvtp1 = []
predictions_tvtp5 = []
actuals_tvtp5 = []
predictions_tvtp22 = []
actuals_tvtp22 = []
# 初始化参数（保持不变，供每次优化使用）
k_features = X_train.shape[1]  # 9个特征
k = k_features + 1  # +1 for constant
n_states = 2
n_params = k * n_states + n_states + 2 * n_states
initial_params = np.zeros(n_params)

# 使用OLS初始化参数
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

# Initialize last_filtered_probs_normalized before the loop
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

        pred_prob = np.clip(pred_prob, 1e-10, 1.0)
        pred_prob /= np.sum(pred_prob)

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


# ======================== 修改后的代码 ========================

print("开始滚动窗口预测 (仅预测1步)...")
# 初始化只用于1步预测的列表
predictions_tvtp1 = []
actuals_tvtp1 = []

for i in tqdm(range(len(X_test)), desc="1-step Ahead Forecast"):
    X_train_vals = rolling_X
    y_train_vals = rolling_y
    z_train_vals = rolling_z

    X_train_with_const = np.column_stack([np.ones(len(X_train_vals)), X_train_vals])

    try:
        # 1. 模型拟合 (这部分保持不变)
        tvtp_fit_result, final_log_filtered_probs_train = fit_tvtp_ms_har(
            y_train_vals, X_train_with_const, z_train_vals, n_states, initial_params=initial_params
        )
        if tvtp_fit_result.success:
            if i % 50 == 0:  # 减少打印频率
                print(f"时间窗 {i}: 模型拟合成功。")
            initial_params = tvtp_fit_result.x
            # 获取并更新最后一步的滤波概率 (这部分也保持不变)
            last_log_probs = final_log_filtered_probs_train[-1, :]
            exp_last_log_probs = np.exp(last_log_probs - np.max(last_log_probs))
            last_filtered_probs_normalized = exp_last_log_probs / np.sum(exp_last_log_probs)
        else:
            if i % 50 == 0:
                print(f"时间窗 {i}: 模型拟合失败。原因: {tvtp_fit_result.message}")

        # 2. 准备预测参数 (这部分保持不变)
        k_with_const = X_train_with_const.shape[1]
        beta_tvtp = initial_params[:k_with_const * n_states].reshape(n_states, k_with_const)
        a_params = initial_params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
        b_params = initial_params[k_with_const * n_states + 2 * n_states:]

        # 3. 进行1步预测 (只保留这部分)
        current_X_test_features = X_test.iloc[i:i + 1].values
        z_for_P_matrix = z_test.iloc[i - 1] if i > 0 else (z_train_vals[-1] if len(z_train_vals) > 0 else 0.0)

        pred_1_log = predict_tvtp_corrected(
            current_X_test_features,
            z_for_P_matrix,
            last_filtered_probs_normalized,
            beta_tvtp,
            a_params,
            b_params,
            n_states
        )
        # 指数还原
        pred_1 = np.exp(pred_1_log)
        actual_1 = np.exp(y_test.iloc[i])

        predictions_tvtp1.append(float(pred_1))
        actuals_tvtp1.append(float(actual_1))

        # 减少打印预测值和实际值的频率，使进度条更清晰
        if i % 50 == 0 or i == len(X_test) - 1:
            print(f"时间窗 {i} - 1步预测值: {pred_1:.6f}, 实际值: {actual_1:.6f} (logRV预测: {pred_1_log:.4f})")

        # [!!!] 删除所有关于 5步 和 22步预测的代码 [!!!]
        # (原先的 pred_5, pred_22, predictions_tvtp5, actuals_tvtp5 等部分全部删除)

    except Exception as e:
        print(f"迭代 {i} 出现错误: {str(e)}")
        # 错误处理也简化，只处理1步预测
        last_pred_1 = predictions_tvtp1[-1] if predictions_tvtp1 else np.nan
        predictions_tvtp1.append(np.exp(last_pred_1))
        actuals_tvtp1.append(float(np.exp(y_test.iloc[i])))

    # 4. 更新滚动窗口 (这部分保持不变)
    new_X = X_test.iloc[i:i + 1].values
    rolling_X = np.vstack((rolling_X[1:], new_X))
    rolling_y = np.append(rolling_y[1:], y_test.iloc[i])
    rolling_z = np.append(rolling_z[1:], z_test.iloc[i])

print("预测完成!")

# 创建结果DataFrame，只包含1步预测结果
df_predictions_tvtp = pd.DataFrame({
    'Prediction_1': predictions_tvtp1,
    'Actual_1': actuals_tvtp1,
})

# 保存结果
df_predictions_tvtp.to_csv('har-cj_corrected_1step.csv', index=False)
print("结果已保存到 'har-cj_corrected_1step.csv'")

# ======================== 修改结束 ========================
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
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
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv
import os
from statsmodels.tsa.vector_ar.var_model import VAR
# Read the data
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
print(all_RV)
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
# 计算各市场对SSE的净溢出效应
npdc_df = npdc_to_btc(gfevd_results, columns)
print(npdc_df)
npdc = npdc_df.iloc[:, 1:-1]
# 选择需要分析的列（排除 SSE 自身和总和列）

# 计算哪个市场对 BTC 溢出最多
npdc_columns = ['NPDC_DASH_BTC', 'NPDC_ETH_BTC', 'NPDC_LTC_BTC', 'NPDC_XLM_BTC', 'NPDC_XRP_BTC']
max_spillover_to_btc = npdc_df[npdc_columns].apply(lambda x: x.idxmax() if x.max() > 0 else None, axis=1)
max_spillover_to_btc_counts = max_spillover_to_btc.value_counts()

# 输出结果
print("=== 哪个市场对 BTC 溢出最多 ===")
print("各市场对 BTC 溢出最多的频率：")
print(max_spillover_to_btc_counts)
print("\n平均 NPDC 值（正值表示对 BTC 的净溢出）：")
print(npdc_df[npdc_columns].mean())




# # 可视化NPDC随时间变化
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 6))
# for col in npdc_columns:
#     plt.plot(npdc_df.index, npdc_df[col], label=col)
# plt.axhline(0, color='black', linestyle='--')
# plt.title('Net Pairwise Directional Connectedness to/from SSE')
# plt.xlabel('Time')
# plt.ylabel('NPDC (%)')
# plt.legend()
# plt.show()

common_columns = all_RV.columns.intersection(npdc.columns)

# 遍历共同列并更新值
data1_modified = all_RV.copy()
print(data1_modified)
#
# # 遍历共同列并更新值
# for col in common_columns:
#     if col != 'Date':  # 跳过 Date 列
#         # 使用 vectorized operation 更新值
#         data1_modified[col] = np.where(npdc[col] < 0, all_RV[col], 0)
#
data1_modified = data1_modified.rename(columns={"BTC": "RV"})
# print(data1_modified)


# 数据准备 (从您提供的代码)
model1 = pd.DataFrame({
    'RV': data1_modified['RV'],
    'rv_lag1': data1_modified['RV'].shift(1),
    'rv_lag5': data1_modified['RV'].rolling(window=5).mean().shift(1),
    'rv_lag22': data1_modified['RV'].rolling(window=22).mean().shift(1),
    'BTC_lag1': data1_modified['LTC'].shift(1)
}).dropna()

test_size = 300

train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

X_train = train_data[['rv_lag1', 'rv_lag5', 'rv_lag22']]
y_train = train_data['RV']
X_test = test_data[['rv_lag1', 'rv_lag5', 'rv_lag22']]
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
            fit_successful_this_iteration = False
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
        a_params = initial_params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
        b_params = initial_params[k_with_const * n_states + 2 * n_states:]

        # 1步预测
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
        pred_1 = pred_1_log  # 直接使用 log(RV)
        actual_1 = y_test.iloc[i]  # 直接使用 log(RV)

        predictions_tvtp1.append(float(pred_1))
        actuals_tvtp1.append(float(actual_1))
        if i % 50 == 0 or i == len(X_test) - 1:
            print(f"时间窗 {i} - 1步预测值: {pred_1:.6f}, 实际值: {actual_1:.6f} (logRV预测: {pred_1_log:.4f})")

        # 5步预测
        if i + 4 < len(X_test):
            future_X_test_5_features = X_test.iloc[i + 4:i + 5].values
            z_for_P_matrix_5 = z_test.iloc[i + 3] if i + 3 < len(z_test) else (
                z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_5_log = predict_tvtp_corrected(
                future_X_test_5_features,
                z_for_P_matrix_5,
                last_filtered_probs_normalized,
                beta_tvtp,
                a_params,
                b_params,
                n_states
            )
            pred_5 = pred_5_log  # 直接使用 log(RV)
            actual_5 = y_test.iloc[i + 4]  # 直接使用 log(RV)

            predictions_tvtp5.append(float(pred_5))
            actuals_tvtp5.append(float(actual_5))
        else:
            predictions_tvtp5.append(None)
            actuals_tvtp5.append(None)

        # 22步预测
        if i + 21 < len(X_test):
            future_X_test_22_features = X_test.iloc[i + 21:i + 22].values
            z_for_P_matrix_22 = z_test.iloc[i + 20] if i + 20 < len(z_test) else (
                z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_22_log = predict_tvtp_corrected(
                future_X_test_22_features,
                z_for_P_matrix_22,
                last_filtered_probs_normalized,
                beta_tvtp,
                a_params,
                b_params,
                n_states
            )
            pred_22 = pred_22_log  # 直接使用 log(RV)
            actual_22 = y_test.iloc[i + 21]  # 直接使用 log(RV)

            predictions_tvtp22.append(float(pred_22))
            actuals_tvtp22.append(float(actual_22))
        else:
            predictions_tvtp22.append(None)
            actuals_tvtp22.append(None)

    except Exception as e:
        print(f"迭代 {i} 出现错误: {str(e)}")
        last_pred_1 = predictions_tvtp1[-1] if predictions_tvtp1 else np.nan
        predictions_tvtp1.append(last_pred_1)
        actuals_tvtp1.append(float(y_test.iloc[i]))  # 直接使用 log(RV)

        last_pred_5 = predictions_tvtp5[-1] if predictions_tvtp5 and predictions_tvtp5[-1] is not None else np.nan
        predictions_tvtp5.append(last_pred_5)
        actual_5_val = float(y_test.iloc[i + 4]) if i + 4 < len(y_test) else np.nan
        actuals_tvtp5.append(actual_5_val)

        last_pred_22 = predictions_tvtp22[-1] if predictions_tvtp22 and predictions_tvtp22[-1] is not None else np.nan
        predictions_tvtp22.append(last_pred_22)
        actual_22_val = float(y_test.iloc[i + 21]) if i + 21 < len(y_test) else np.nan
        actuals_tvtp22.append(actual_22_val)

    new_X = X_test.iloc[i:i + 1].values
    rolling_X = np.vstack((rolling_X[1:], new_X))
    rolling_y = np.append(rolling_y[1:], y_test.iloc[i])
    rolling_z = np.append(rolling_z[1:], z_test.iloc[i])
print("预测完成!")

# 创建结果DataFrame，现在包含还原后的RV值
df_predictions_tvtp = pd.DataFrame({
    'Prediction_1': predictions_tvtp1,
    'Actual_1': actuals_tvtp1,
    'Prediction_5': predictions_tvtp5,
    'Actual_5': actuals_tvtp5,
    'Prediction_22': predictions_tvtp22,
    'Actual_22': actuals_tvtp22
})

# 保存结果
df_predictions_tvtp.to_csv('har-rv_corrected.csv', index=False)
print("结果已保存到 'har-rv_corrected.csv'")

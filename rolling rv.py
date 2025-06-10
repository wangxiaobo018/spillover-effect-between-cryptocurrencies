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
    returns = (prices / prices.shift(1) - 1)*100
    returns.iloc[0] = 0  # First return is 0
    returns[prices.shift(1) == 0] = np.nan  # Handle division by zero
    return returns

# Calculate returns by group
data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)

# Get group summary for data_ret
group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')

# List of cryptocurrency IDs to process
crypto_ids = [ "000001.XSHG", "SH.000016","SH.000300","SZ.399001","SZ.399006"]

# Initialize an empty DataFrame to store all RV values
all_RV = pd.DataFrame()

# Loop through each cryptocurrency ID
for crypto_id in crypto_ids:
    # Filter for the current cryptocurrency and remove unnecessary columns
    data_filtered_crypto = data_ret[data_ret['id'] == crypto_id].copy()
    data_filtered_crypto = data_filtered_crypto.drop('id', axis=1)

    # Convert DT to datetime and calculate daily RV
    data_filtered_crypto['DT'] = pd.to_datetime(data_filtered_crypto['DT']).dt.date

    RV_crypto = (data_filtered_crypto
                 .groupby('DT')['Ret']
                 .apply(lambda x: np.sum(x**2))
                 .reset_index())

    # Ensure RV has the correct column names
    RV_crypto.columns = ['DT', crypto_id]

    # Convert DT to datetime for consistency
    RV_crypto['DT'] = pd.to_datetime(RV_crypto['DT'])

    # Merge the current RV with the all_RV DataFrame
    if all_RV.empty:
        all_RV = RV_crypto
    else:
        all_RV = pd.merge(all_RV, RV_crypto, on='DT', how='outer')

# Sort the final DataFrame by date
all_RV = all_RV.sort_values(by='DT')
all_RV= all_RV.copy()
all_RV.columns = ["DT"] + crypto_ids  # 保持与 crypto_ids 一致
all_RV.columns = ["DT", "BTC", "DASH", "ETH", "LTC", "XLM"]


# Extract features and convert to numpy array
features = all_RV.drop(columns=['DT'])
data = features.to_numpy()

# Define the column names to match the renamed columns (excluding DT)
columns = ["BTC", "DASH", "ETH", "LTC", "XLM"]  # Use the desired names instead of crypto_ids
data_df = pd.DataFrame(data, columns=columns)

# Update NPDC and RV column names
npdc_columns = [f"NPDC_{col}_BTC" for col in columns if col != "BTC"] + ["NPDC_BTC_BTC"]
rv_columns = columns[1:]  # Exclude "BTC" as the reference market

# Parameters
n = len(columns)  # Number of variables
p = 1  # Lag order
H = 12  # Forecast horizon
kappa1 = 0.99  # Forgetting factor 1
kappa2 = 0.96  # Forgetting factor 2
T = data.shape[0]  # Time length

# Initialize TVP-VAR parameters
var_model = VAR(data_df[:3]).fit(maxlags=p)
A_0 = np.hstack([coef for coef in var_model.coefs])  # Initial coefficient matrix
Sigma_0 = np.cov(data[:3], rowvar=False)  # Initial covariance matrix
Sigma_A_0 = np.eye(n * p) * 0.1  # Initial coefficient covariance

# TVP-VAR model (Kalman filter)
def tvp_var(y, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0):
    T, n = y.shape
    A_t = np.zeros((n, n * p, T))  # Time-varying coefficients
    Sigma_t = np.zeros((n, n, T))  # Time-varying covariance
    Sigma_A_t = np.zeros((n * p, n * p, T))  # Coefficient covariance

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

# Run TVP-VAR
A_t, Sigma_t, Sigma_A_t = tvp_var(data, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0)

# Calculate Generalized Forecast Error Variance Decomposition (GFEVD)
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

# Calculate GFEVD
gfevd_results = gfevd(A_t, Sigma_t, H, n)

# Calculate NPDC to BTC
def npdc_to_btc(gfevd_array, columns):
    T = gfevd_array.shape[2]
    n = len(columns)
    npdc_btc = np.zeros((T, n))

    for t in range(T):
        for j in range(n):
            npdc_btc[t, j] = (gfevd_array[0, j, t] - gfevd_array[j, 0, t]) * 100  # Assuming BTC is the 0th column

    npdc_df = pd.DataFrame(npdc_btc, columns=[f"NPDC_{col}_BTC" for col in columns])
    other_cols = [f"NPDC_{col}_BTC" for col in columns if col != 'BTC']
    npdc_df['Total_NPDC_others_BTC'] = npdc_df[other_cols].sum(axis=1)
    return npdc_df

# Calculate total spillover
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

# Calculate net spillover to BTC
npdc_df = npdc_to_btc(gfevd_results, columns)
print(npdc_df)

# Select columns for analysis (exclude BTC self and total)
npdc = npdc_df.iloc[:, 1:-1]

# Calculate which market spills over to BTC the most
npdc_columns = ['NPDC_DASH_BTC', 'NPDC_ETH_BTC', 'NPDC_LTC_BTC', 'NPDC_XLM_BTC']
max_spillover_to_btc = npdc_df[npdc_columns].apply(lambda x: x.idxmax() if x.max() > 0 else None, axis=1)
max_spillover_to_btc_counts = max_spillover_to_btc.value_counts()

# Output results
print("=== Which market spills over to BTC the most ===")
print("Frequency of each market having the highest spillover to BTC:")
print(max_spillover_to_btc_counts)
print("\nAverage NPDC values (positive values indicate net spillover to BTC):")
print(npdc_df[npdc_columns].mean())


#
# def adjust_rv_by_max_spillover(npdc_df, rv_df,
#                                npdc_columns=['NPDC_SSE50_SSE', 'NPDC_CSI_SSE', 'NPDC_SZCZ_SSE', 'NPDC_STAR_SSE'],
#                                rv_columns=['SSE50', 'CSI', 'SZCZ', 'STAR']):
#     """
#     根据 NPDC 值调整已实现波动率：
#     - 在时间点 t，若 NPDC_CSI_SSE 不是最大的正值，则选择其他市场（SSE50, SZCZ, STAR）中正值最大的市场，
#       用该市场的 RV 替换 RV_CSI。
#     - 如果没有市场有正的 NPDC 值，保留 RV_CSI。
#
#     参数：
#     - npdc_df: 包含 NPDC 值的 DataFrame，列包括 NPDC_SSE50_SSE, NPDC_CSI_SSE, NPDC_SZCZ_SSE, NPDC_STAR_SSE 等。
#     - rv_df: 包含已实现波动率的 DataFrame，列包括 SSE, SSE50, CSI, SZCZ, STAR, DT。
#     - npdc_columns: NPDC 列名列表，默认为 ['NPDC_SSE50_SSE', 'NPDC_CSI_SSE', 'NPDC_SZCZ_SSE', 'NPDC_STAR_SSE']。
#     - rv_columns: RV 列名列表，默认为 ['SSE50', 'CSI', 'SZCZ', 'STAR']，对应 npdc_columns。
#
#     返回：
#     - adjusted_rv_df: 包含调整后 RV 的 DataFrame，新增 RV_CSI_adjusted 列。
#     """
#     # 验证输入数据
#     assert npdc_df.shape[0] == rv_df.shape[0], "npdc_df 和 rv_df 的行数必须一致"
#     assert all(col in npdc_df.columns for col in npdc_columns), "npdc_df 缺少指定的 NPDC 列"
#     assert all(col in rv_df.columns for col in rv_columns), "rv_df 缺少指定的 RV 列"
#
#     # 创建调整后的 RV DataFrame
#     adjusted_rv_df = rv_df.copy()
#     adjusted_rv_df['RV_CSI_adjusted'] = adjusted_rv_df['CSI']
#
#     # 遍历每个时间点
#     for t in npdc_df.index:
#         # 获取当前时间点的 NPDC 值
#         npdc_values = npdc_df.loc[t, npdc_columns]
#
#         # 找到正值最大的市场
#         positive_npdc = npdc_values[npdc_values > 0]
#         if not positive_npdc.empty:
#             max_market = positive_npdc.idxmax()
#             if max_market != 'NPDC_CSI_SSE':
#                 # 如果 CSI 不是最大正值市场，用最大正值市场的 RV 替换 RV_CSI
#                 market_idx = npdc_columns.index(max_market)
#                 rv_col = rv_columns[market_idx]
#                 adjusted_rv_df.loc[t, 'RV_CSI_adjusted'] = rv_df.loc[t, rv_col]
#
#     return adjusted_rv_df
#
# # 运行调整函数
# adjusted_rv_df = adjust_rv_by_max_spillover(
#     npdc_df,
#     all_RV,
#     npdc_columns=['NPDC_SSE50_SSE', 'NPDC_CSI_SSE', 'NPDC_SZCZ_SSE', 'NPDC_STAR_SSE'],
#     rv_columns=['SSE50', 'CSI', 'SZCZ', 'STAR']
# )


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
    'BTC_lag1': data1_modified['ETH'].shift(1)
}).dropna()

''''
test_size = 500

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

df_predictions_tvtp.to_csv('tvtphar-rv_corrected_DASH.csv', index=False)
print("结果已保存到 'tvtphar-rv_corrected_DASH.csv'")

'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
import uuid

# Define window and test sizes
window_size = 1000  # Rolling window size
test_size = 500     # Test set size

# Split training and test data
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

df_predictions_tvtp.to_csv('tvtphar-rv_window.csv', index=False)
print("结果已保存到 'tvtphar-rv_window.csv'")
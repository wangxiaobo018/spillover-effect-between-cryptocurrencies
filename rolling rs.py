import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution, minimize
import datetime
import statsmodels.tools.numdiff as nd
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
# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/result.csv")

# Get group summary
group_summary = df.groupby('code').size().reset_index(name='NumObservations')

# Create data_ret DataFrame with renamed columns first
data_ret = df[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()


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
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure both 'DT' columns are datetime
RV['DT'] = pd.to_datetime(RV['DT'])
print(RV)
def calculate_returns(prices):
    """Calculate returns from price series."""
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            returns[i] = np.nan
        else:
            returns[i] = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
    returns[0] = 0
    return returns


def calculate_RS(data):
    """Calculate RS+ and RS- from returns."""
    positive_returns = np.where(data['Ret'] > 0, data['Ret'], 0)
    negative_returns = np.where(data['Ret'] < 0, data['Ret'], 0)

    RS_plus = np.sum(np.square(positive_returns))
    RS_minus = np.sum(np.square(negative_returns))

    return pd.Series({
        'RS_plus': RS_plus,
        'RS_minus': RS_minus
    })


def process_har_rs_model(data_idx_path, index_list):
    """Process data for HAR-RS model for multiple indices."""
    # Read the index data
    df_idx = pd.read_csv(data_idx_path)
    print(f"原始数据形状: {df_idx.shape}")

    # 检查所需指数是否都在数据中
    available_indices = df_idx['code'].unique()
    print(f"数据中的指数: {available_indices}")

    missing_indices = [idx for idx in index_list if idx not in available_indices]
    if missing_indices:
        print(f"警告: 以下指数在数据中不存在: {missing_indices}")

    # 过滤只保留需要的指数
    df_idx = df_idx[df_idx['code'].isin(index_list)]
    print(f"过滤后数据形状: {df_idx.shape}")

    # 检查每个指数的数据量
    index_counts = df_idx['code'].value_counts()
    print("每个指数的记录数:")
    print(index_counts)

    # Process index data
    data_ret_idx = (
        df_idx[['time', 'code', 'close']]
        .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
        .dropna()
    )

    # Calculate returns for index data
    grouped_idx = data_ret_idx.groupby('id')
    returns_list_idx = []

    for name, group in grouped_idx:
        # 确保按时间排序
        group = group.sort_values('DT')
        group_returns = pd.DataFrame({
            'DT': group['DT'],
            'id': group['id'],
            'Ret': calculate_returns(group['PRICE'].values)
        })
        returns_list_idx.append(group_returns)

    data_ret_idx = pd.concat(returns_list_idx, ignore_index=True)

    # 添加日期列用于分组
    data_ret_idx['Date'] = pd.to_datetime(data_ret_idx['DT']).dt.date

    # 计算每个指数每天的RS统计
    results = []

    # 按指数和日期分组
    for (idx_id, date), group in data_ret_idx.groupby(['id', 'Date']):
        # 计算RS统计
        rs_values = calculate_RS(group)
        # 添加结果
        results.append({
            'id': idx_id,
            'Date': date,
            'DT': group['DT'].iloc[0],  # 保存原始DT值，用于最终输出
            'RS_plus': rs_values['RS_plus'],
            'RS_minus': rs_values['RS_minus']
        })

    # 转换为DataFrame
    result_df = pd.DataFrame(results)

    # 检查每个指数的日期覆盖范围
    print("\n每个指数的日期范围:")
    for idx in index_list:
        if idx in result_df['id'].values:
            idx_data = result_df[result_df['id'] == idx]
            print(f"{idx}: {idx_data['Date'].min()} 至 {idx_data['Date'].max()}, 共 {len(idx_data)} 天")

    return result_df


def process_final_output(result_df, code_map):
    """处理最终输出，将RS数据转换为宽格式，并保留日期列"""
    # 确保Date是日期类型
    result_df['Date'] = pd.to_datetime(result_df['Date'])

    # 映射指数代码
    result_df['id_mapped'] = result_df['id'].map(code_map)

    # 检查是否有未映射的指数
    unmapped = result_df['id_mapped'].isna()
    if unmapped.any():
        print(f"警告: {unmapped.sum()} 行数据的指数未被映射")
        print(f"未映射的指数: {result_df.loc[unmapped, 'id'].unique()}")
        # 删除未映射的行
        result_df = result_df.dropna(subset=['id_mapped'])

    # 使用pivot_table并保留DT列
    # 先按Date分组，找到每个日期对应的一个DT值
    dt_by_date = result_df.groupby('Date')['DT'].first().reset_index()

    # 创建RS+和RS-的透视表
    rs_plus = pd.pivot_table(
        result_df,
        index='Date',
        columns='id_mapped',
        values='RS_plus',
        aggfunc='sum'  # 如果有重复项，则求和
    )

    rs_minus = pd.pivot_table(
        result_df,
        index='Date',
        columns='id_mapped',
        values='RS_minus',
        aggfunc='sum'  # 如果有重复项，则求和
    )

    # 确保所有需要的列都存在
    for col in ['SSE', 'SSE50', 'CSI', 'SZCZ', 'STAR']:
        if col not in rs_plus.columns:
            rs_plus[col] = np.nan
        if col not in rs_minus.columns:
            rs_minus[col] = np.nan

    # 重新排序列，并添加DT
    rs_plus = rs_plus.reset_index()
    rs_minus = rs_minus.reset_index()

    # 合并DT列
    rs_plus = pd.merge(rs_plus, dt_by_date, on='Date', how='left')
    rs_minus = pd.merge(rs_minus, dt_by_date, on='Date', how='left')

    # 调整列顺序
    rs_plus = rs_plus[['DT', 'SSE', 'SSE50', 'CSI', 'SZCZ', 'STAR']]
    rs_minus = rs_minus[['DT', 'SSE', 'SSE50', 'CSI', 'SZCZ', 'STAR']]

    return rs_plus, rs_minus


# 使用示例:
if __name__ == "__main__":
    data_idx_path = Path("c:/Users/lenovo/Desktop/spillover/result.csv")
    # 要计算的指数列表
    index_list = ["000001.XSHG", "SH.000016", "SH.000300", "SZ.399001", "SZ.399006"]

    # 指数代码映射
    code_map = {
        "000001.XSHG": "SSE",
        "SH.000016": "SSE50",
        "SH.000300": "CSI",
        "SZ.399001": "SZCZ",
        "SZ.399006": "STAR"
    }

    print("开始处理数据...")
    final_data = process_har_rs_model(data_idx_path, index_list)

    print("\n最终数据样本:")
    print(final_data.head(10))

    # 检查数据中的指数
    print(f"\n数据中的指数: {final_data['id'].unique()}")

    # 生成最终输出
    print("\n生成最终输出...")
    rs_p, rs_m = process_final_output(final_data, code_map)

    print("\nRS+ 数据框:")
    print(rs_p.head())
    print(f"RS+ 列名: {rs_p.columns.tolist()}")

    print("\nRS- 数据框:")
    print(rs_m.head())
    print(f"RS- 列名: {rs_m.columns.tolist()}")



from statsmodels.tsa.vector_ar.var_model import VAR

# Ensure datasets are defined
try:
    datasets = {
        'all_RS': rs_p,
        'all_RM': rs_m,
    }
except NameError:
    raise NameError("Datasets all_RD, all_RM, all_RP must be defined.")

# Parameters
columns = ["SSE", "SSE50", "CSI", "SZCZ", "STAR"]
n = len(columns)
p = 1
H = 12
kappa1 = 0.99
kappa2 = 0.96
npdc_columns = ['NPDC_SSE50_SSE', 'NPDC_CSI_SSE', 'NPDC_SZCZ_SSE', 'NPDC_STAR_SSE']


# TVP-VAR model (Kalman filter)
def tvp_var(y, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0):
    T, n = y.shape
    A_t = np.zeros((n, n * p, T))
    Sigma_t = np.zeros((n, n, T))
    Sigma_A_t = np.zeros((n * p, n * p, T))

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


# Compute GFEVD
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
        diag_sigma = np.diag(Sigma_t[:, :, t])
        if np.any(diag_sigma <= 0):
            diag_sigma = np.maximum(diag_sigma, 1e-10)
        for h in range(H):
            Psi_t[:, :, h] = B_jt[:, :, h] @ Sigma_t[:, :, t] @ np.diag(1 / np.sqrt(diag_sigma))

        gfevd_t = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gfevd_t[i, j] = np.sum(Psi_t[i, j, :] ** 2) / np.sum(Psi_t[i, :, :] ** 2)
        gfevd_t = gfevd_t / gfevd_t.sum(axis=1, keepdims=True)
        gfevd_array[:, :, t] = gfevd_t

    return gfevd_array


# Compute NPDC to SSE
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


# Process each dataset
for dataset_name, dataset in datasets.items():
    print(f"\n=== Processing {dataset_name} ===")

    # Data preparation
    try:
        dataset.columns = ["DT", "SSE", "SSE50", "CSI", "SZCZ", "STAR"]
    except ValueError:
        print(f"Error: {dataset_name} does not have the expected number of columns.")
        continue

    features = dataset.drop(columns=['DT'])
    data = features.to_numpy()
    data_df = pd.DataFrame(data, columns=columns)

    T = data.shape[0]
    if T < 3:
        print(f"Error: {dataset_name} has insufficient data (T={T}). Skipping.")
        continue

    # Initialize TVP-VAR parameters
    try:
        var_model = VAR(data_df[:3]).fit(maxlags=p)
        A_0 = np.hstack([coef for coef in var_model.coefs])
        Sigma_0 = np.cov(data[:3], rowvar=False)
        if np.any(np.isnan(Sigma_0)):
            Sigma_0 = np.eye(n)
        Sigma_A_0 = np.eye(n * p) * 0.1
    except Exception as e:
        print(f"Error fitting VAR for {dataset_name}: {e}. Skipping.")
        continue

    # Run TVP-VAR
    A_t, Sigma_t, Sigma_A_t = tvp_var(data, p, kappa1, kappa2, A_0, Sigma_0, Sigma_A_0)

    # Compute GFEVD
    gfevd_results = gfevd(A_t, Sigma_t, H, n)

    # Compute NPDC to SSE
    npdc_df = npdc_to_sse(gfevd_results, columns)

    # Compute which market spills over to SSE the most
    max_spillover_to_sse = npdc_df[npdc_columns].apply(lambda x: x.idxmax() if x.max() > 0 else None, axis=1)
    max_spillover_to_sse_counts = max_spillover_to_sse.value_counts()

    # Print requested results
    print(f"\n=== 哪个市场对SSE溢出最多 ({dataset_name}) ===")
    print("各市场对SSE溢出最多的频率：")
    print(max_spillover_to_sse_counts)
    print("\n平均NPDC值（正值表示对SSE的净溢出）：")
    print(npdc_df[npdc_columns].mean())



# 创建一个新的 CSI 已实现波动率列，初始值为原始 CSI 列
rs_p['SZCZ_reconstructed'] = rs_p['SZCZ'].copy()

# 定义需要比较的市场
markets_to_compare = ['CSI', 'STAR']
npdc_columns_to_compare = [f'NPDC_{market}_SSE' for market in markets_to_compare]

# 遍历每个时间点
for t in range(len(rs_p)):
    # 检查 CSI 的净溢出值是否小于 0
    if npdc_df['NPDC_CSI_SSE'].iloc[t] < 0:
        # 获取 STAR 和 SZCZ 的净溢出值
        npdc_values = {}
        for market, npdc_col in zip(markets_to_compare, npdc_columns_to_compare):
            npdc_value = npdc_df[npdc_col].iloc[t]
            # 仅考虑非 NaN 且大于 0 的净溢出值
            if not np.isnan(npdc_value) and npdc_value > 0:
                npdc_values[market] = npdc_value

        # 如果有符合条件的净溢出值（大于 0）
        if npdc_values:
            # 找到净溢出值最大的市场
            max_market = max(npdc_values, key=npdc_values.get)
            # 用该市场的已实现波动率替换 CSI 的值
            rs_p.at[t, 'SZCZ_reconstructed'] = rs_p[max_market].iloc[t]


rs_p_lag1 = rs_p['SSE'].shift(1)
rs_m_lag1 = rs_m['SSE'].shift(1)
rs_p_lag5 =rs_p['SSE'].rolling(window=5).mean().shift(1)
rs_m_lag5 = rs_m['SSE'].rolling(window=5).mean().shift(1)
rs_p_lag22 = rs_p['SSE'].rolling(window=22).mean().shift(1)
rs_m_lag22 = rs_m['SSE'].rolling(window=22).mean().shift(1)

model1 = pd.DataFrame({
    'RV': np.log(RV['Ret']),
    'rs_p_lag1': rs_p_lag1,
    'rs_m_lag1': rs_m_lag1,
    'rs_p_lag5': rs_p_lag5,
    'rs_m_lag5': rs_m_lag5,
    'rs_p_lag22': rs_p_lag22,
    'rs_m_lag22': rs_m_lag22,
    'trans1': rs_p['SZCZ'].diff(1)
}).dropna()

test_size = 300

train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

X_train = train_data[['rs_p_lag1', 'rs_m_lag1', 'rs_p_lag5', 'rs_m_lag5', 'rs_p_lag22', 'rs_m_lag22']]
y_train = train_data['RV']
X_test = test_data[['rs_p_lag1', 'rs_m_lag1', 'rs_p_lag5', 'rs_m_lag5', 'rs_p_lag22', 'rs_m_lag22']]
y_test = test_data['RV']
z_train = train_data['trans1']
z_test = test_data['trans1']

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
df_predictions_tvtp.to_csv('har-rs_corrected_1step.csv', index=False)
print("结果已保存到 'har-rs_corrected_1step.csv'")
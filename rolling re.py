from statsmodels.tsa.vector_ar.var_model import VAR
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
from statsmodels.tools.eval_measures import aic, bic, hqic
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
import statsmodels.tools.numdiff as nd # <--- 添加这一行

from scipy.stats import norm


# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/result.csv")


SSE= df[df['code'] == "000001.XSHG"].copy()

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

# Calculate returns for each group using the new formula
def calculate_returns(prices):
    # Compute daily returns using the given formula
    returns = (prices / prices.shift(1) - 1) * 100  # (Pt - Pt-1) / Pt-1 * 100
    returns.iloc[0] = 0  # First return is 0
    returns[prices.shift(1) == 0] = np.nan  # Handle division by zero
    return returns

sse = get_re(SSE, alpha=0.05)

# RV

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
data_filtered = data_ret[data_ret['id'] == "000001.XSHG"].copy()
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


# -------------


# Define the code map
code_map = {
    "000001.XSHG": "SSE",
    "SH.000016": "SSE50",
    "SH.000300": "CSI",
    "SZ.399001": "SZCZ",
    "SZ.399006": "STAR"
}

# Calculate all_RM (REX_minus)
ct_dfs = []
for code, col_name in code_map.items():
    stock_df = df[df['code'] == code].copy()
    result = get_re(stock_df, alpha=0.05)
    # Include 'day' and 'REX_minus', rename 'day' to 'DT' and 'REX_minus' to col_name
    ct_df = result[['day', 'REX_minus']].rename(columns={'day': 'DT', 'REX_minus': col_name})
    ct_dfs.append(ct_df)

# Merge all results on 'DT'
all_RM = ct_dfs[0][['DT']]  # Start with the DT column from the first DataFrame
for ct_df in ct_dfs:
    all_RM = all_RM.merge(ct_df[['DT', ct_df.columns[1]]], on='DT', how='outer')

# Rename columns to the desired format
all_RM.columns = ['DT', 'SSE', 'SSE50', 'CSI', 'SZCZ', 'STAR']
# Convert DT to datetime for consistency
all_RM['DT'] = pd.to_datetime(all_RM['DT'])

# Calculate all_RP (REX_plus)
ct_dfs = []
for code, col_name in code_map.items():
    stock_df = df[df['code'] == code].copy()
    result = get_re(stock_df, alpha=0.05)
    # Include 'day' and 'REX_plus', rename 'day' to 'DT' and 'REX_plus' to col_name
    ct_df = result[['day', 'REX_plus']].rename(columns={'day': 'DT', 'REX_plus': col_name})
    ct_dfs.append(ct_df)

# Merge all results on 'DT'
all_RP = ct_dfs[0][['DT']]  # Start with the DT column from the first DataFrame
for ct_df in ct_dfs:
    all_RP = all_RP.merge(ct_df[['DT', ct_df.columns[1]]], on='DT', how='outer')

# Rename columns to the desired format
all_RP.columns = ['DT', 'SSE', 'SSE50', 'CSI', 'SZCZ', 'STAR']
# Convert DT to datetime for consistency
all_RP['DT'] = pd.to_datetime(all_RP['DT'])

# Calculate all_RD (REX_moderate)
ct_dfs = []
for code, col_name in code_map.items():
    stock_df = df[df['code'] == code].copy()
    result = get_re(stock_df, alpha=0.05)
    # Include 'day' and 'REX_moderate', rename 'day' to 'DT' and 'REX_moderate' to col_name
    ct_df = result[['day', 'REX_moderate']].rename(columns={'day': 'DT', 'REX_moderate': col_name})
    ct_dfs.append(ct_df)

# Merge all results on 'DT'
all_RD = ct_dfs[0][['DT']]  # Start with the DT column from the first DataFrame
for ct_df in ct_dfs:
    all_RD = all_RD.merge(ct_df[['DT', ct_df.columns[1]]], on='DT', how='outer')

# Rename columns to the desired format
all_RD.columns = ['DT', 'SSE', 'SSE50', 'CSI', 'SZCZ', 'STAR']
# Convert DT to datetime for consistency
all_RD['DT'] = pd.to_datetime(all_RD['DT'])


import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR

# Ensure datasets are defined
try:
    datasets = {
        'all_RD': all_RD,
        'all_RM': all_RM,
        'all_RP': all_RP
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

assert len(all_RD) == len(npdc_df), "all_RD 和 npdc_df 的长度必须一致"

# 创建一个新的 SZCZ 已实现波动率列，初始值为原始 SZCZ 列
all_RD['SZCZ_reconstructed'] = all_RD['SZCZ'].copy()

# 定义需要比较的市场
markets_to_compare = ['STAR', 'CSI']
npdc_columns_to_compare = [f'NPDC_{market}_SSE' for market in markets_to_compare]

# 遍历每个时间点
for t in range(len(all_RD)):
    # 检查 SZCZ 的净溢出值是否小于 0
    if npdc_df['NPDC_SZCZ_SSE'].iloc[t] < 0:
        # 获取 STAR 和 CSI 的净溢出值
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
            # 用该市场的已实现波动率替换 SZCZ 的值
            all_RD.at[t, 'SZCZ_reconstructed'] = all_RD[max_market].iloc[t]


# 创建一个新的 CSI 已实现波动率列，初始值为原始 CSI 列
all_RD['SZCZ_reconstructed'] = all_RP['SZCZ'].copy()

# 定义需要比较的市场
markets_to_compare = ['STAR', 'CSI']
npdc_columns_to_compare = [f'NPDC_{market}_SSE' for market in markets_to_compare]

# 遍历每个时间点
for t in range(len(all_RD)):
    # 检查 CSI 的净溢出值是否小于 0
    if npdc_df['NPDC_SZCZ_SSE'].iloc[t] < 0:
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
            all_RD.at[t, 'SZCZ_reconstructed'] = all_RD[max_market].iloc[t]



# 创建一个新的 CSI 已实现波动率列，初始值为原始 CSI 列
all_RM['CSI_re'] = all_RM['CSI'].copy()

# 定义需要比较的市场
markets_to_compare = ['SZCZ']
npdc_columns_to_compare = ['NPDC_SZCZ_SSE']

# 遍历每个时间点
for t in range(len(all_RM)):
    # 检查 SZCZ 的净溢出值是否小于 0
    if npdc_df['NPDC_CSI_SSE'].iloc[t] < 0:
        # 获取 STAR 的净溢出值
        npdc_value = npdc_df['NPDC_SZCZ_SSE'].iloc[t]
        # 仅考虑非 NaN 且大于 0 的净溢出值
        if not np.isnan(npdc_value) and npdc_value > 0:
            # 用 STAR 市场的已实现波动率替换 SZCZ 的值
            all_RM.at[t, 'CSI_re'] = all_RM['SZCZ'].iloc[t]


# 创建一个新的 CSI 已实现波动率列，初始值为原始 CSI 列
all_RP['CSI_reconstructed'] = all_RP['CSI'].copy()

# 定义需要比较的市场
markets_to_compare = ['STAR', 'SZCZ']
npdc_columns_to_compare = [f'NPDC_{market}_SSE' for market in markets_to_compare]

# 遍历每个时间点
for t in range(len(all_RP)):
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
            all_RP.at[t, 'CSI_reconstructed'] = all_RP[max_market].iloc[t]


rex_m_lag1 = all_RM['SSE'].shift(1)
rex_m_lag5 =all_RM['SSE'].rolling(window=5).mean().shift(1)
rex_m_lag22 = all_RM['SSE'].rolling(window=22).mean().shift(1)
rex_p_lag1 = all_RP['SSE'].shift(1)
rex_p_lag5 = all_RP['SSE'].rolling(window=5).mean().shift(1)
rex_p_lag22 = all_RP['SSE'].rolling(window=22).mean().shift(1)
rex_md_lag1 = all_RD['SSE'].shift(1)
rex_md_lag5 = all_RD['SSE'].rolling(window=5).mean().shift(1)
rex_md_lag22 = all_RD['SSE'].rolling(window=22).mean().shift(1)


# 数据准备 (从您新提供的数据结构)
model1 = pd.DataFrame({
    'RV': np.log(RV['RV']),
    'REX_m_lag1': rex_m_lag1,
    'REX_m_lag5': rex_m_lag5,
    'REX_m_lag22': rex_m_lag22,
    'REX_p_lag1': rex_p_lag1,
    'REX_p_lag5': rex_p_lag5,
    'REX_p_lag22': rex_p_lag22,
    'REX_md_lag1': rex_md_lag1,
    'REX_md_lag5': rex_md_lag5,
    'REX_md_lag22': rex_md_lag22,
    'trans3': all_RD['SZCZ'].shift(1)
}).dropna()

test_size = 300

train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

# 修改特征变量，使用新的RE相关特征
X_train = train_data[['REX_m_lag1', 'REX_m_lag5', 'REX_m_lag22',
                      'REX_p_lag1', 'REX_p_lag5', 'REX_p_lag22',
                      'REX_md_lag1', 'REX_md_lag5', 'REX_md_lag22']]
y_train = train_data['RV']
X_test = test_data[['REX_m_lag1', 'REX_m_lag5', 'REX_m_lag22',
                    'REX_p_lag1', 'REX_p_lag5', 'REX_p_lag22',
                    'REX_md_lag1', 'REX_md_lag5', 'REX_md_lag22']]
y_test = test_data['RV']
z_train = train_data['trans3']  # 现在使用trans3作为转换概率的影响因子
z_test = test_data['trans3']

rolling_X = X_train.values
rolling_y = y_train.values
rolling_z = z_train.values

predictions_tvtp1 = []
actuals_tvtp1 = []
predictions_tvtp5 = []
actuals_tvtp5 = []
predictions_tvtp22 = []
actuals_tvtp22 = []

k_features = X_train.shape[1]  # 现在有9个特征而不是3个
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
df_predictions_tvtp.to_csv('har-re_corrected_1step.csv', index=False)
print("结果已保存到 'har-re_corrected_1step.csv'")
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

from statsmodels.regression.quantile_regression import QuantReg

# 定义文件路径字典
data_files = {
    'BTC': "BTCUSDT_5m.csv",
    'DASH': "DASHUSDT_5m.csv",
    'ETH': "ETHUSDT_5m.csv",
    'LTC': "LTCUSDT_5m.csv",
    'XLM': "XLMUSDT_5m.csv",
    'XRP': "XRPUSDT_5m.csv"
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

features = all_CT.drop(columns=['DT'])
data = features.to_numpy()

columns = ["BTC", "DASH", "ETH", "LTC", "XLM", "XRP"]
data_df = pd.DataFrame(data, columns=columns)
##  选择2最优
# 步骤1: 计算 BTC 波动率的 0.05 和 0.95 分位数作为阈值
rv_quantiles = data_df['BTC'].quantile([0.05, 0.95])

# 步骤2: 使用 pd.cut 根据阈值创建 'market_state' 列
data_df['market_state'] = pd.cut(data_df['BTC'],
                                 bins=[-np.inf, rv_quantiles[0.05], rv_quantiles[0.95], np.inf],
                                 labels=['Bear', 'Normal', 'Bull'])

# 步骤3: 使用嵌套的 np.where 构建动态驱动变量
data_df['Dynamic_RV_lag1'] = np.where(
    data_df['market_state'] == 'Bear',
    data_df['XRP'],
    np.where(
        data_df['market_state'] == 'Bull',
        data_df['XRP'],
        data_df['XRP']
    )
)

final_output = data_df[['BTC', 'market_state', 'LTC', 'XLM', 'XRP', 'Dynamic_RV_lag1']].dropna()

# Read the data
df_data = pd.read_csv("BTCUSDT_5m.csv")

# Get group summary
group_summary = df_data.groupby('code').size().reset_index(name='NumObservations')

# Create data_ret DataFrame with renamed columns first
data_ret = df_data[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()


# Calculate returns for each group using the new formula
def calculate_returns(prices):
    # Compute daily returns using the given formula
    returns = (prices / prices.shift(1) - 1) * 100
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
      .apply(lambda x: np.sum(x ** 2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']
print(RV)
# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])

JV_lag1 = all_JV['BTC'].shift(1)
C_t_lag1 = all_CT['BTC'].shift(1)
JV_lag5 = all_JV['BTC'].rolling(window=5).mean().shift(1)
C_t_lag5 = all_CT['BTC'].rolling(window=5).mean().shift(1)
JV_lag22 = all_JV['BTC'].rolling(window=22).mean().shift(1)
C_t_lag22 = all_JV['BTC'].rolling(window=22).mean().shift(1)

# 数据准备

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_hess

# 数据准备
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_hess  # 正确的导入方式

# Data preparation
model_data = pd.DataFrame({
    'RV': RV['RV'],
    'Jv_lag1': JV_lag1,
    'Jv_lag5': JV_lag5,
    'Jv_lag22': JV_lag22,
    'C_t_lag1': C_t_lag1,
    'C_t_lag5': C_t_lag5,
    'C_t_lag22': C_t_lag22,
    'BTC_lag1': final_output['Dynamic_RV_lag1'].shift(1)
}).dropna()

train_size = int(len(model_data) * 0.8)
train_data = model_data.iloc[:train_size]


def ms_har_time_varying_log_likelihood(params, y, X, Z, n_states=2):
    """
    时变转移概率MS-HAR模型的对数似然函数
    """
    n = len(y)
    k = X.shape[1]

    # 解析参数
    beta = params[:k * n_states].reshape(n_states, k)
    sigma = np.exp(params[k * n_states:k * n_states + n_states])
    alpha_00, gamma_00, alpha_11, gamma_11 = params[k * n_states + n_states:]

    log_filtered_prob = np.zeros((n, n_states))
    log_lik = 0.0

    # 预计算均值
    mu_cache = np.zeros((n, n_states))
    for j in range(n_states):
        mu_cache[:, j] = X @ beta[j]

    for t in range(n):
        # 计算时变转移概率
        logit_p00_t = alpha_00 + gamma_00 * Z[t]
        logit_p11_t = alpha_11 + gamma_11 * Z[t]
        p00_t = np.clip(1.0 / (1.0 + np.exp(-logit_p00_t)), 1e-5, 1 - 1e-5)
        p11_t = np.clip(1.0 / (1.0 + np.exp(-logit_p11_t)), 1e-5, 1 - 1e-5)
        P_t = np.array([[p00_t, 1.0 - p00_t], [1.0 - p11_t, p11_t]])

        # 预测概率
        if t == 0:
            pred_prob = np.ones(n_states) / n_states
        else:
            filtered_prob_prev = np.exp(log_filtered_prob[t - 1])
            pred_prob = filtered_prob_prev @ P_t

        pred_prob = np.clip(pred_prob, 1e-10, 1.0)

        # 计算条件密度
        log_conditional_densities = np.zeros(n_states)
        for j in range(n_states):
            log_conditional_densities[j] = norm.logpdf(y[t], mu_cache[t, j], sigma[j])

        # 计算联合概率和边际概率
        log_joint_prob = np.log(pred_prob) + log_conditional_densities

        # 数值稳定性处理
        if np.all(np.isinf(log_joint_prob)):
            return 1e10

        max_log_prob = np.max(log_joint_prob)
        log_marginal_prob = max_log_prob + np.log(np.sum(np.exp(log_joint_prob - max_log_prob)))

        log_filtered_prob[t] = log_joint_prob - log_marginal_prob
        log_lik += log_marginal_prob

    if np.isnan(log_lik) or np.isinf(log_lik):
        return 1e10
    return -log_lik


def compute_hessian_and_std_errors(params, y, X, Z, n_states=2):
    """
    使用Hessian方法计算标准误差
    """
    print("\nComputing Hessian matrix...")

    # 计算Hessian矩阵
    try:
        # 使用approx_hess函数
        hessian = approx_hess(params, ms_har_time_varying_log_likelihood,
                              args=(y, X, Z, n_states))

        print("Hessian matrix computed successfully")

        # 检查Hessian的条件数
        cond_number = np.linalg.cond(hessian)
        print(f"Hessian condition number: {cond_number:.2e}")

        if cond_number > 1e10:
            print("Warning: Hessian is poorly conditioned!")

        # 检查特征值
        eigenvalues = np.linalg.eigvals(hessian)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        print(f"Eigenvalue range: [{min_eig:.2e}, {max_eig:.2e}]")

        # 如果Hessian不是正定的，进行修正
        if min_eig <= 0:
            print("Warning: Hessian is not positive definite. Adding regularization...")
            # 添加正则化项使其正定
            regularization = max(1e-6, -min_eig * 1.1)
            hessian_reg = hessian + regularization * np.eye(len(params))
            print(f"Added regularization: {regularization:.2e}")
        else:
            hessian_reg = hessian

        # 计算协方差矩阵
        try:
            cov_matrix = np.linalg.inv(hessian_reg)
            print("Covariance matrix computed successfully")

            # 提取标准误差
            var_diag = np.diag(cov_matrix)
            if np.any(var_diag < 0):
                print("Warning: Negative variances detected!")
                std_errors = np.sqrt(np.abs(var_diag))
            else:
                std_errors = np.sqrt(var_diag)

            return std_errors, cov_matrix, True

        except np.linalg.LinAlgError as e:
            print(f"Failed to invert Hessian: {str(e)}")
            # 尝试使用伪逆
            print("Trying pseudo-inverse...")
            cov_matrix = np.linalg.pinv(hessian_reg)
            std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
            return std_errors, cov_matrix, False

    except Exception as e:
        print(f"Error computing Hessian: {str(e)}")
        n_params = len(params)
        return np.full(n_params, np.nan), np.full((n_params, n_params), np.nan), False


def fit_ms_har_time_varying(y, X, Z, var_names, n_states=2):
    """
    拟合具有时变转移概率的MS-HAR模型（使用Hessian方法）
    """
    k = X.shape[1]
    n_params = k * n_states + n_states + 4

    # 初始化参数
    print("Initializing parameters...")
    initial_params = np.zeros(n_params)

    # OLS初始化
    ols_model = sm.OLS(y, X).fit()
    for s in range(n_states):
        factor = 0.6 + 0.8 * s
        initial_params[s * k:(s + 1) * k] = ols_model.params * factor + np.random.normal(0, 0.05, k)

    # 初始化sigma
    residuals = y - X @ ols_model.params
    sigma_base = np.std(residuals)
    for s in range(n_states):
        initial_params[k * n_states + s] = np.log(sigma_base * (0.5 + s))

    # 初始化转移参数
    initial_params[k * n_states + n_states:] = [1.5, 0.1, 1.5, -0.1]

    # 设置边界
    bounds = []
    for _ in range(k * n_states):
        bounds.append((None, None))
    for _ in range(n_states):
        bounds.append((np.log(1e-4), np.log(5 * np.std(y))))
    bounds.extend([(-5, 5), (-2, 2), (-5, 5), (-2, 2)])

    # 优化
    print("\nStarting optimization...")
    try:
        result = minimize(
            ms_har_time_varying_log_likelihood,
            initial_params,
            args=(y, X, Z, n_states),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 20000}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        raise ValueError("Optimization attempt failed")

    # 使用Hessian方法计算标准误差
    std_errors, cov_matrix, success = compute_hessian_and_std_errors(
        result.x, y, X, Z, n_states
    )

    if not success:
        print("\nWarning: Standard errors may be unreliable due to numerical issues.")

    # 计算统计量
    t_stats = result.x / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

    # 打印结果
    print("\nParameter estimates with standard errors and significance:")
    param_names = []
    for s in range(n_states):
        for var in var_names:
            param_names.append(f"beta_{s}_{var}")
    for s in range(n_states):
        param_names.append(f"sigma_{s}")
    param_names.extend(['alpha_00', 'gamma_00', 'alpha_11', 'gamma_11'])

    for i, (name, param, se, t_stat, p_val) in enumerate(
            zip(param_names, result.x, std_errors, t_stats, p_values)
    ):
        if np.isnan(se):
            print(f"{name}: {param:.6f} (SE: nan, t-stat: nan, p-value: nan)")
        else:
            print(f"{name}: {param:.6f} (SE: {se:.6f}, t-stat: {t_stat:.4f}, p-value: {p_val:.4f})")

    return result, std_errors, t_stats, p_values, cov_matrix


# 主函数
def main():
    # 准备数据
    y = train_data['RV'].values
    har_vars = ['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']
    X_har = train_data[har_vars].values
    X = np.column_stack([np.ones(len(X_har)), X_har])
    var_names_with_const = ['const'] + har_vars
    Z = train_data['BTC_lag1'].values

    print("Fitting time-varying MS-HAR-CJ model...")
    try:
        result = fit_ms_har_time_varying(y, X, Z, var_names_with_const)
        ms_result, std_errors, t_stats, p_values, cov_matrix = result
        print("\nTime-varying MS-HAR-CJ model fitted successfully!")

        # 输出结果摘要
        k = X.shape[1]
        n_states = 2
        beta_ms = ms_result.x[:k * n_states].reshape(n_states, k)
        sigma_ms = np.exp(ms_result.x[k * n_states:k * n_states + n_states])

        print("\n--- Time-varying MS-HAR-CJ Results (Standardized Scale) ---")

        # 确定高低波动状态
        low_vol_state = np.argmin(sigma_ms)
        high_vol_state = np.argmax(sigma_ms)

        print(f"\nLow Volatility State (State {low_vol_state}):")
        for i in range(k):
            print(
                f"  {var_names_with_const[i]:<15}: {beta_ms[low_vol_state, i]:.6f} (SE: {std_errors[low_vol_state * k + i]:.6f})")
        print(f"  {'sigma':<15}: {sigma_ms[low_vol_state]:.6f} (SE: {std_errors[k * n_states + low_vol_state]:.6f})")

        print(f"\nHigh Volatility State (State {high_vol_state}):")
        for i in range(k):
            print(
                f"  {var_names_with_const[i]:<15}: {beta_ms[high_vol_state, i]:.6f} (SE: {std_errors[high_vol_state * k + i]:.6f})")
        print(f"  {'sigma':<15}: {sigma_ms[high_vol_state]:.6f} (SE: {std_errors[k * n_states + high_vol_state]:.6f})")

        print("\n--- Transition Probability Parameters (Interpretation) ---")
        print("Parameters estimated on standardized Z (BTC_lag1):")
        alpha_00_idx = k * n_states + n_states
        print(f"  alpha_00_std: {ms_result.x[alpha_00_idx]:.4f}, gamma_00_std: {ms_result.x[alpha_00_idx + 1]:.4f}")
        print(f"  alpha_11_std: {ms_result.x[alpha_00_idx + 2]:.4f}, gamma_11_std: {ms_result.x[alpha_00_idx + 3]:.4f}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
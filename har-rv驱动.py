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

# 参数设置
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




# 变量列表
variables = ['RV', 'rv_lag1', 'rv_lag5', 'rv_lag22', 'BTC_lag1']


# Improved t-Copula implementation
class CustomTCopula:
    def __init__(self, dim):
        self.dim = dim
        self.correlation = None
        self.df = None  # degrees of freedom

    def fit(self, data):
        """Fit t-copula to uniform data"""
        # Convert to normal scores
        normal_scores = norm.ppf(data)

        # Estimate correlation matrix
        self.correlation = np.corrcoef(normal_scores, rowvar=False)

        # Estimate degrees of freedom (simplified approach)
        # For a more accurate approach, you'd use MLE
        self.df = 5  # Starting with a common value for financial data
        return self

    def pdf(self, u_point):
        """Calculate the density function of the t-copula at point u"""
        if not hasattr(self, 'correlation') or not hasattr(self, 'df'):
            raise ValueError("Copula must be fitted before calling pdf")

        # Convert uniform variables to t quantiles
        t_point = t.ppf(u_point, self.df)

        # Get dimension from point
        dim = len(t_point)

        # Calculate multivariate t density
        det_corr = np.linalg.det(self.correlation)
        inv_corr = np.linalg.inv(self.correlation)

        # Quadratic form (x^T Σ^-1 x)
        quad_form = np.dot(np.dot(t_point, inv_corr), t_point)

        # Multivariate t density
        numerator = gamma((self.df + dim) / 2) * det_corr ** (-0.5)
        denominator = gamma(self.df / 2) * (np.pi * self.df) ** (dim / 2) * (1 + quad_form / self.df) ** (
                    (self.df + dim) / 2)
        mvt_density = numerator / denominator

        # Calculate product of marginal densities
        marginal_densities = np.prod([t.pdf(x, self.df) for x in t_point])

        # Copula density is ratio of joint density to product of marginals
        copula_density = mvt_density / marginal_densities

        return copula_density

    def kendall_tau(self):
        """Calculate Kendall's tau from correlation matrix"""
        # For t-copula: tau = (2/π) * arcsin(ρ)
        tau_matrix = np.zeros_like(self.correlation)
        n = self.correlation.shape[0]

        for i in range(n):
            for j in range(n):
                if i != j:
                    tau_matrix[i, j] = 2 * np.arcsin(self.correlation[i, j]) / np.pi
                else:
                    tau_matrix[i, j] = 1.0

        return tau_matrix


# 主分析函数
def run_copula_analysis(model1, variables):
    # 提取数据
    data = model1[variables].values

    # 步骤 1: 转换为均匀分布 (使用经验 CDF)
    u_data = np.array([pd.Series(col).rank() / (len(col) + 1) for col in data.T]).T

    # 步骤 2: 拟合 t-Copula
    copula_t = CustomTCopula(dim=len(variables))
    copula_t.fit(u_data)

    # 步骤 3: 提取 t-Copula 的 Kendall's Tau
    kendall_tau_t = copula_t.kendall_tau()
    print("\nCopula Kendall's Tau (t-Copula):")
    print(pd.DataFrame(kendall_tau_t, columns=variables, index=variables))

    # 步骤 4: 计算皮尔逊相关系数
    pearson_corr = model1[variables].corr(method='pearson')
    print("\nPearson Correlation:")
    print(pearson_corr)

    # 步骤 5: 可视化 Copula 密度 (RV vs BTC_lag1)
    u1 = u_data[:, 0]  # RV
    u2 = u_data[:, 4]  # BTC_lag1

    # 拟合 2D t-Copula 用于可视化
    copula_t_2d = CustomTCopula(dim=2)
    copula_t_2d.fit(np.column_stack([u1, u2]))

    # 创建网格
    u1_grid, u2_grid = np.meshgrid(np.linspace(0.01, 0.99, 100), np.linspace(0.01, 0.99, 100))
    grid_points = np.column_stack([u1_grid.flatten(), u2_grid.flatten()])

    # 计算密度值 (使用向量化操作来提高效率)
    density_values = np.zeros(len(grid_points))

    # 由于pdf计算可能比较耗时，我们可以批量处理或使用采样点
    # 这里为了简化，我们采用每个点单独计算
    for i, point in enumerate(grid_points):
        try:
            density_values[i] = copula_t_2d.pdf(point)
        except (ValueError, np.linalg.LinAlgError):
            # 处理可能的数值问题
            density_values[i] = np.nan

    # 重塑为网格形式
    density_grid = density_values.reshape(u1_grid.shape)

    # 可视化
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(u1_grid, u2_grid, density_grid, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Density')
    plt.scatter(u1, u2, alpha=0.5, s=10, c='red')
    plt.title('t-Copula Density (RV vs BTC_lag1)')
    plt.xlabel('RV (Uniform)')
    plt.ylabel('BTC_lag1 (Uniform)')
    plt.tight_layout()
    plt.show()

    return copula_t, u_data

copula_t, u_data = run_copula_analysis(model1, variables)

def tvtp_ms_har_log_likelihood(params, y, X, z, n_states=2):
    """
    Numerically stable log-likelihood function for TVTP-MS-HAR model
    """
    n = len(y)
    k = X.shape[1]

    # Extract parameters
    beta = params[:k * n_states].reshape(n_states, k)
    sigma = np.exp(params[k * n_states:k * n_states + n_states])
    a = params[k * n_states + n_states:k * n_states + 2 * n_states]
    b = params[k * n_states + 2 * n_states:]

    # Initialize arrays in log space for numerical stability
    log_filtered_prob = np.zeros((n, n_states))
    log_pred_prob = np.ones(n_states) * (-np.log(n_states))  # Uniform initial

    log_lik = 0.0
    # Cache X @ beta for efficiency
    mu_cache = np.zeros((n, n_states))
    for j in range(n_states):
        mu_cache[:, j] = X @ beta[j]

    for t in range(n):
        # Calculate transition probabilities
        P = np.zeros((n_states, n_states))
        if t > 0:
            # Transition probabilities with bounds
            logit_11 = np.clip(a[0] + b[0] * z[t - 1], -30, 30)  # 扩大裁剪范围
            p11 = 1.0 / (1.0 + np.exp(-logit_11))
            logit_22 = np.clip(a[1] + b[1] * z[t - 1], -30, 30)  # 扩大裁剪范围
            p22 = 1.0 / (1.0 + np.exp(-logit_22))
            p11 = np.clip(p11, 0.001, 0.999)
            p22 = np.clip(p22, 0.001, 0.999)
            P[0, 0] = p11
            P[0, 1] = 1.0 - p11
            P[1, 0] = 1.0 - p22
            P[1, 1] = p22
            # Predicted probabilities
            if t == 1:
                pred_prob = np.ones(n_states) / n_states @ P
            else:
                filtered_prob_prev = np.exp(log_filtered_prob[t - 1])
                filtered_prob_prev = filtered_prob_prev / np.sum(filtered_prob_prev)
                pred_prob = filtered_prob_prev @ P
        else:
            pred_prob = np.ones(n_states) / n_states

        pred_prob = np.clip(pred_prob, 1e-10, 1.0)
        # Conditional densities
        log_conditional_densities = np.zeros(n_states)
        for j in range(n_states):
            log_conditional_densities[j] = norm.logpdf(y[t], mu_cache[t, j], sigma[j])

        # Joint log probabilities
        log_joint_prob = np.log(pred_prob) + log_conditional_densities
        # Log-sum-exp trick
        max_log_prob = np.max(log_joint_prob)
        log_marginal_prob = max_log_prob + np.log(np.sum(np.exp(log_joint_prob - max_log_prob)))
        # Update filtered probabilities
        log_filtered_prob[t] = log_joint_prob - log_marginal_prob
        log_lik += log_marginal_prob

    if np.isnan(log_lik) or np.isinf(log_lik):
        return 1e10
    return -log_lik


def fit_tvtp_ms_har(y, X, z, n_states=2):
    """
    Fit TVTP-MS-HAR model without standardizing z
    """
    k = X.shape[1]
    n_params = k * n_states + n_states + 2 * n_states

    # No standardization for z anymore
    # Just print z statistics for diagnostic purposes
    print(
        f"Z stats - Mean: {np.mean(z):.4f}, Std: {np.std(z):.4f}, Min: {np.min(z):.4f}, Max: {np.max(z):.4f}")

    # OLS-based initialization
    initial_params = np.zeros(n_params)
    ols_model = sm.OLS(y, X).fit()

    print("Starting optimization...")

    # HAR parameters initialization
    for s in range(n_states):
        factor = 0.6 + 0.8 * s
        initial_params[s * k:(s + 1) * k] = ols_model.params * factor + np.random.normal(0, 0.05, k)

    # Volatility parameters
    residuals = y - X @ ols_model.params
    sigma_base = np.std(residuals)
    for s in range(n_states):
        initial_params[k * n_states + s] = np.log(sigma_base * (0.5 + s))

    # Transition parameters - using the optimal initialization approach
    # Second initialization scheme from original code
    initial_params[k * n_states + n_states:k * n_states + 2 * n_states] = [0.8, 0.8]
    initial_params[k * n_states + 2 * n_states:] = [-0.1, 0.1]

    # Parameter bounds - adjusted for non-standardized z
    bounds = []
    # Beta parameters bounds
    for _ in range(k * n_states):
        bounds.append((None, None))

    # Sigma parameters bounds
    for _ in range(n_states):
        bounds.append((np.log(0.0001), np.log(5 * np.std(y))))

    # a parameters bounds - expanded
    for _ in range(n_states):
        bounds.append((-7, 7))  # Wider range

    # b parameters bounds - adjusted for non-standardized z
    z_std = np.std(z)
    # The new bounds should be roughly equivalent to the old bounds divided by z_std
    for _ in range(n_states):
        # Adjust based on the scale of z
        if abs(np.mean(z)) < 0.1 and z_std < 0.1:
            # For very small z values, provide wider bounds
            bounds.append((-50, 50))
        else:
            # Scale based on z standard deviation
            bound_scale = min(7 / max(0.1, z_std), 50)  # Limit to reasonable values
            bounds.append((-bound_scale, bound_scale))

    print(f"Parameters: {len(initial_params)}, Bounds: {len(bounds)}")
    assert len(initial_params) == len(bounds)

    try:
        result = minimize(
            tvtp_ms_har_log_likelihood,
            initial_params,
            args=(y, X, z, n_states),  # Use original z directly
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 10000}
        )

        print(f"Converged: {result.success}, Iterations: {result.nit}, Function evals: {result.nfev}")

        # Check for boundary solutions
        for i, (param, bound) in enumerate(zip(result.x, bounds)):
            if bound[0] is not None and np.isclose(param, bound[0], atol=1e-4):
                print(f"WARNING: Parameter {i} at lower bound: {param:.6f} ≈ {bound[0]}")
            if bound[1] is not None and np.isclose(param, bound[1], atol=1e-4):
                print(f"WARNING: Parameter {i} at upper bound: {param:.6f} ≈ {bound[1]}")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        raise ValueError("Optimization attempt failed")

    print(f"TVTP-MS-HAR optimization succeeded: NLL = {result.fun:.4f}")

    # Analyze the transition probabilities given the fitted parameters
    a_params = result.x[k * n_states + n_states:k * n_states + 2 * n_states]
    b_params = result.x[k * n_states + 2 * n_states:]

    # Calculate transition probabilities at different z values
    # Use actual z range instead of standardized range
    z_min, z_max = np.min(z), np.max(z)
    z_values = np.linspace(z_min, z_max, 5)  # Range of original z values
    print("\nTransition probabilities at different z values:")
    print("z value | P(1→1) | P(1→2) | P(2→1) | P(2→2)")
    for z_val in z_values:
        p11 = 1 / (1 + np.exp(-(a_params[0] + b_params[0] * z_val)))
        p22 = 1 / (1 + np.exp(-(a_params[1] + b_params[1] * z_val)))
        print(f"{z_val:7.4f} | {p11:6.4f} | {1 - p11:6.4f} | {1 - p22:6.4f} | {p22:6.4f}")

    return result, None  # Return None for scaler_z since we're not using it anymore

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic, hqic
from pandas import DataFrame
from scipy.stats import norm


def run_model_estimation(model1):
    """
    Execute TVTP-MS-HAR model estimation with improved diagnostics and statsmodels-based inference
    """
    print("Preparing data...")
    y = model1['RV'].values

    # 先获取不含截距的特征矩阵
    X_no_const = model1[['rv_lag1', 'rv_lag5', 'rv_lag22']].values

    # 为TVTP-MS-HAR模型添加截距
    X = np.column_stack([np.ones(len(X_no_const)), X_no_const])

    z = model1['BTC_lag1'].values

    # 线性模型比较也使用相同的X（已包含截距）
    X_sm = X

    # 数据诊断部分
    print(f"\nData summary:")
    print(f"Sample size: {len(y)}")
    print(f"RV mean: {np.mean(y):.6f}, std: {np.std(y):.6f}, min: {np.min(y):.6f}, max: {np.max(y):.6f}")
    print(f"BTC_lag1 mean: {np.mean(z):.6f}, std: {np.std(z):.6f}")

    if np.any(np.isnan(y)) or np.any(np.isnan(X)) or np.any(np.isnan(z)):
        print("WARNING: NaN values detected in the data!")

    # 更新变量名和相关性计算
    vars_with_const = ['const', 'rv_lag1', 'rv_lag5', 'rv_lag22']
    print("\nCorrelations with RV:")
    for i, var in enumerate(vars_with_const):
        if i > 0:  # 跳过截距的相关性计算
            print(f"{var}: {np.corrcoef(y, X[:, i])[0, 1]:.4f}")
    print(f"BTC_lag1: {np.corrcoef(y, z)[0, 1]:.4f}")

    # Fit linear HAR model for comparison using statsmodels
    print("\nFitting linear HAR model for comparison...")
    har_model = sm.OLS(y, X_sm).fit()

    # ============ 修正的线性HAR模型AIC/BIC计算 ============
    n = len(y)
    # 计算线性HAR模型的对数似然值（不是负对数似然值）
    linear_log_likelihood = -n / 2 * (1 + np.log(2 * np.pi * har_model.mse_resid))  # 注意这里是负数
    linear_n_params = X_sm.shape[1] + 1  # 参数数量（回归系数 + sigma）

    # 使用对数似然值计算AIC/BIC
    linear_aic = aic(linear_log_likelihood, linear_n_params, n)
    linear_bic = bic(linear_log_likelihood, linear_n_params, n)

    print("\nFitting TVTP-MS-HAR model...")
    try:
        tvtp_result, _ = fit_tvtp_ms_har(y, X, z)
        print("\nTVTP-MS-HAR model fitted successfully!")
    except Exception as e:
        print(f"\nTVTP-MS-HAR fitting failed: {str(e)}")
        print("Using linear HAR model as fallback:")
        print(har_model.summary())
        return None

    # Extract parameters
    k = X.shape[1]  # 现在包含截距，所以k增加了1
    n_states = 2
    beta_tvtp = tvtp_result.x[:k * n_states].reshape(n_states, k)
    sigma_tvtp = np.exp(tvtp_result.x[k * n_states:k * n_states + n_states])
    a_tvtp = tvtp_result.x[k * n_states + n_states:k * n_states + n_states * 2]
    b_tvtp = tvtp_result.x[k * n_states + n_states * 2:]

    # Display results
    print("\nTVTP-MS-HAR results:")
    states = ["Low volatility state", "High volatility state"]
    vars = vars_with_const  # 使用包含截距的变量名列表

    for s in range(n_states):
        print(f"\n{states[s]}:")
        for i in range(k):
            print(f"  {vars[i]}: {beta_tvtp[s, i]:.6f}")
        print(f"  sigma: {sigma_tvtp[s]:.6f}")

    print("\nTransition probability parameters:")
    for s in range(n_states):
        print(f"  a_{s + 1}: {a_tvtp[s]:.6f}, b_{s + 1}: {b_tvtp[s]:.6f}")

    # Calculate standard errors using statsmodels information criteria approach
    print("\nCalculating statistical significance using statsmodels...")

    try:
        # Use statsmodels' numeric Hessian computation
        import statsmodels.tools.numdiff as nd

        # Function for computing the negative log-likelihood (for a single point)
        def nll_function(params):
            return tvtp_ms_har_log_likelihood(params, y, X, z, n_states)

        # Compute Hessian at the minimum
        H = nd.approx_hess(tvtp_result.x, nll_function)

        # Compute covariance matrix
        cov_matrix = np.linalg.inv(H)
        std_errors = np.sqrt(np.diag(cov_matrix))

        # Create parameter names for better reporting
        param_names = []
        for s in range(n_states):
            for i, var in enumerate(vars):
                param_names.append(f"β_{s + 1}_{var}")
        for s in range(n_states):
            param_names.append(f"σ_{s + 1}")
        for s in range(n_states):
            param_names.append(f"a_{s + 1}")
        for s in range(n_states):
            param_names.append(f"b_{s + 1}")

        # Create a DataFrame for nice reporting
        results_df = DataFrame({
            'Parameter': param_names,
            'Estimate': tvtp_result.x,
            'Std Error': std_errors,
            't-value': tvtp_result.x / std_errors,
            'p-value': 2 * (1 - norm.cdf(abs(tvtp_result.x / std_errors)))
        })

        # Add significance stars
        results_df['Significance'] = ''
        results_df.loc[results_df['p-value'] < 0.1, 'Significance'] = '*'
        results_df.loc[results_df['p-value'] < 0.05, 'Significance'] += '*'
        results_df.loc[results_df['p-value'] < 0.01, 'Significance'] += '*'

        print("\nParameter Estimates and Statistical Significance:")
        print(results_df)

        # Special focus on transition parameters
        print("\nBTC收益对转移概率的影响:")
        b_indices = [k * n_states + 2 * n_states, k * n_states + 2 * n_states + 1]

        for s, idx in enumerate(b_indices):
            b_val = tvtp_result.x[idx]
            se = std_errors[idx]
            t_stat = b_val / se
            p_val = 2 * (1 - norm.cdf(abs(t_stat)))
            significance = "显著" if p_val < 0.05 else "不显著"
            effect = "正向" if b_val > 0 else "负向"
            print(f"  b_{s + 1}: {b_val:.6f} (SE: {se:.6f}, t: {t_stat:.2f}, p: {p_val:.4f}, {significance})")
            print(f"  解释: BTC收益对状态{s + 1}的持续性有{effect}影响" + (", 但影响不显著" if p_val >= 0.05 else ""))

    except Exception as e:
        print(f"\nError computing standard errors: {str(e)}")
        print("Using optimization-based standard errors instead...")
        try:
            # Original approach as fallback
            if hasattr(tvtp_result, 'hess_inv'):
                if isinstance(tvtp_result.hess_inv, np.ndarray):
                    hess_inv = tvtp_result.hess_inv
                else:
                    hess_inv = tvtp_result.hess_inv.todense()

                std_errors = np.sqrt(np.diag(hess_inv))

                for i, name in enumerate(param_names):
                    val = tvtp_result.x[i]
                    se = std_errors[i]
                    t_stat = val / se
                    p_val = 2 * (1 - norm.cdf(abs(t_stat)))
                    significance = "*" * sum([p_val < th for th in [0.1, 0.05, 0.01]])
                    if not significance:
                        significance = "not significant"
                    print(f"  {name}: {val:.6f} (SE: {se:.6f}, t: {t_stat:.2f}, p: {p_val:.4f}) {significance}")
            else:
                print("\nHessian not available. Cannot compute standard errors and p-values.")
        except:
            print("\nFailed to compute standard errors using fallback method.")

    # ============ 修正的TVTP-MS-HAR模型AIC/BIC计算 ============
    n_params_tvtp = len(tvtp_result.x)
    # tvtp_result.fun是负对数似然值，所以取负号得到对数似然值
    log_likelihood = -tvtp_result.fun  # 对数似然值（通常是负数）

    # 使用对数似然值计算AIC/BIC（不是负对数似然值）
    aic_value = aic(log_likelihood, n_params_tvtp, n)
    bic_value = bic(log_likelihood, n_params_tvtp, n)
    hqic_value = hqic(log_likelihood, n_params_tvtp, n)

    print("\nTVTP-MS-HAR Model Information Criteria:")
    print(f"Log-likelihood: {log_likelihood:.4f}")  # 对数似然值（负数）
    print(f"Negative Log-likelihood: {-log_likelihood:.4f}")  # 负对数似然值（正数）
    print(f"Parameters: {n_params_tvtp}, Sample size: {n}")
    print(f"AIC: {aic_value:.4f}")  # 现在应该是正数
    print(f"BIC: {bic_value:.4f}")  # 现在应该是正数
    print(f"HQIC: {hqic_value:.4f}")  # 现在应该是正数

    # ============ 修正的模型比较 ============
    linear_metrics = {
        'Log-likelihood': linear_log_likelihood,  # 对数似然值
        'AIC': linear_aic,
        'BIC': linear_bic
    }

    print("\nModel comparison:")
    print("Linear HAR model metrics:")
    for metric, value in linear_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nImprovement from TVTP-MS-HAR over linear HAR:")
    print(f"  AIC improvement: {linear_aic - aic_value:.4f}")  # 正数表示TVTP更好
    print(f"  BIC improvement: {linear_bic - bic_value:.4f}")  # 正数表示TVTP更好

    # Transition probability interpretation
    z_std = np.std(z)
    print("\n参数解释:")
    for s in range(n_states):
        p_base = 1.0 / (1.0 + np.exp(-a_tvtp[s]))
        p_up = 1.0 / (1.0 + np.exp(-(a_tvtp[s] + b_tvtp[s] * z_std)))
        p_down = 1.0 / (1.0 + np.exp(-(a_tvtp[s] - b_tvtp[s] * z_std)))

        direction = "增加" if b_tvtp[s] > 0 else "减少"
        print(f"  状态{s + 1}的基础持续概率: {p_base:.4f}")
        print(f"  当BTC收益为+1σ({z_std:.6f})时, 状态{s + 1}的持续概率: {p_up:.4f}")
        print(f"  当BTC收益为-1σ({-z_std:.6f})时, 状态{s + 1}的持续概率: {p_down:.4f}")
        print(f"  BTC收益上升时, 状态{s + 1}的持续概率{direction}")

    return tvtp_result, har_model

def main(model1):
    """
    Main function to run TVTP-MS-HAR analysis
    """
    print("Starting TVTP-MS-HAR volatility model analysis...")
    try:
        tvtp_result = run_model_estimation(model1)
        if tvtp_result is not None:
            print("\nAnalysis completed successfully.")
            print("\n建议: 如果模型参数接近边界，可能需要考虑以下几点:")
            print("1. 进一步扩大参数约束范围")
            print("2. 检查数据质量和预处理步骤")
            print("3. 考虑不同的模型规范，如增加状态数或调整转移变量")
        else:
            print("\nModel estimation failed, falling back to linear model.")
    except Exception as e:
        print(f"\nERROR: Analysis failed: {str(e)}")
        print("\nRecommendation: Check data quality or simplify model.")


if __name__ == "__main__":
    print("\n========== Running TVTP-MS-HAR Analysis ==========\n")
    main(model1)


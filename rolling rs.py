import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import norm

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

def calculate_returns(prices):
    """Calculate returns from price series."""
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            returns[i] = np.nan
        else:
            returns[i] = ((prices[i] - prices[i - 1]) / prices[i - 1])*100
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

def process_har_rs_model(data_idx_path, crypto_id):
    """Process data for HAR-RS model for a single cryptocurrency."""
    # Read the index data
    df_idx = pd.read_csv(data_idx_path)

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
        group_returns = pd.DataFrame({
            'DT': group['DT'],
            'id': group['id'],
            'Ret': calculate_returns(group['PRICE'].values)
        })
        returns_list_idx.append(group_returns)

    data_ret_idx = pd.concat(returns_list_idx, ignore_index=True)

    # Filter for specific index
    data_cj = data_ret_idx.query(f'id == "{crypto_id}"').copy()

    # Calculate RS statistics by date
    result = (
        data_cj.groupby(pd.to_datetime(data_cj['DT']).dt.date)
        .apply(calculate_RS)
        .reset_index()
    )

    # Rename columns for consistency
    result = result.rename(columns={'level_0': 'DT'})
    return result

def combine_rs_data(data_files):
    """Combine RS_plus and RS_minus for multiple cryptocurrencies."""
    rs_plus_list = []
    rs_minus_list = []

    for crypto_id, file_path in data_files.items():
        # Process each cryptocurrency
        result = process_har_rs_model(file_path, crypto_id)
        # Convert DT to datetime for consistency
        result['DT'] = pd.to_datetime(result['DT'])
        # Create DataFrames for RS_plus and RS_minus
        rs_plus = result[['DT', 'RS_plus']].rename(columns={'RS_plus': crypto_id})
        rs_minus = result[['DT', 'RS_minus']].rename(columns={'RS_minus': crypto_id})
        rs_plus_list.append(rs_plus)
        rs_minus_list.append(rs_minus)

    # Merge all RS_plus DataFrames on DT
    rs_plus_combined = rs_plus_list[0]
    for df in rs_plus_list[1:]:
        rs_plus_combined = rs_plus_combined.merge(df, on='DT', how='outer')

    # Merge all RS_minus DataFrames on DT
    rs_minus_combined = rs_minus_list[0]
    for df in rs_minus_list[1:]:
        rs_minus_combined = rs_minus_combined.merge(df, on='DT', how='outer')

    # Set DT as index
    rs_plus_combined.set_index('DT', inplace=True)
    rs_minus_combined.set_index('DT', inplace=True)

    return rs_plus_combined, rs_minus_combined

# Example usage
if __name__ == "__main__":
    data_files = {
        'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv",
        'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/dash.csv",
        'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/eth.csv",
        'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ltc.csv",
        'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/xlm.csv",
        'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/xrp.csv"
    }

    # Process all cryptocurrencies and combine results
    rs_plus_df, rs_minus_df = combine_rs_data(data_files)

    print("\nRS_plus Combined Data Sample:")
    print(rs_plus_df.head())
    print("\nRS_minus Combined Data Sample:")
    print(rs_minus_df.head())

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

rs_p_lag1 = rs_plus_df['BTC'].shift(1)
rs_p_lag5 = rs_plus_df['BTC'].rolling(window=5).mean().shift(1)
rs_p_lag22 = rs_plus_df['BTC'].rolling(window=22).mean().shift(1)
rs_m_lag1 = rs_minus_df['BTC'].shift(1)
rs_m_lag5 = rs_minus_df['BTC'].rolling(window=5).mean().shift(1)
rs_m_lag22 = rs_minus_df['BTC'].rolling(window=22).mean().shift(1)

model1= pd.DataFrame({
    'RV': RV,
    'rs_p_lag1': rs_p_lag1,
    'rs_p_lag5': rs_p_lag5,
    'rs_p_lag22': rs_p_lag22,
    'rs_m_lag1': rs_m_lag1,
    'rs_m_lag5': rs_m_lag5,
    'rs_m_lag22': rs_m_lag22,
    'BTC_lag1':rs_plus_df['ETH'].shift(1)
}).dropna()

'''
test_size = 300

train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]


X_train = train_data[['rs_p_lag1', 'rs_p_lag5', 'rs_p_lag22', 'rs_m_lag1', 'rs_m_lag5', 'rs_m_lag22']]
y_train = train_data['RV']
X_test = test_data[['rs_p_lag1', 'rs_p_lag5', 'rs_p_lag22', 'rs_m_lag1', 'rs_m_lag5', 'rs_m_lag22']]
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

df_predictions_tvtp.to_csv('tvtphar-rs_corrected_DASH.csv', index=False)
print("结果已保存到 'tvtphar-rs_corrected_DASH.csv'")
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize

# Set new parameters
window_size = 1000
test_size = 500

# Data splitting
train_data = model1.iloc[:len(model1) - test_size]
test_data = model1.iloc[len(model1) - test_size:]

X_train = train_data[['rs_p_lag1', 'rs_p_lag5', 'rs_p_lag22', 'rs_m_lag1', 'rs_m_lag5', 'rs_m_lag22']]
y_train = train_data['RV']
X_test = test_data[['rs_p_lag1', 'rs_p_lag5', 'rs_p_lag22', 'rs_m_lag1', 'rs_m_lag5', 'rs_m_lag22']]
y_test = test_data['RV']
z_train = train_data['BTC_lag1']
z_test = test_data['BTC_lag1']

# Initialize rolling window with window_size
rolling_X = X_train[-window_size:].values  # Take last window_size rows
rolling_y = y_train[-window_size:].values
rolling_z = z_train[-window_size:].values

# Initialize lists for predictions and actuals
predictions_tvtp1 = []
actuals_tvtp1 = []
predictions_tvtp5 = []
actuals_tvtp5 = []
predictions_tvtp22 = []
actuals_tvtp22 = []

# Model parameters
k_features = X_train.shape[1]
k = k_features + 1  # Including intercept
n_states = 2
n_params = k * n_states + n_states + 2 * n_states
initial_params = np.zeros(n_params)

# Initialize parameters using OLS
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

# Initialize last_filtered_probs_normalized
last_filtered_probs_normalized = np.ones(n_states) / n_states

# Define tvtp_ms_har_log_likelihood function (unchanged)
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

# Define fit_tvtp_ms_har function (unchanged)
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

# Define predict_tvtp_corrected function (unchanged)
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

# Rolling window predictions
print("开始滚动窗口预测...")
for i in tqdm(range(len(X_test))):
    X_train_vals = rolling_X[-window_size:]  # Ensure window_size
    y_train_vals = rolling_y[-window_size:]
    z_train_vals = rolling_z[-window_size:]

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
            last_log_probs = final_log_filtered_probs_train[-1, :]
            exp_last_log_probs = np.exp(last_log_probs - np.max(last_log_probs))
            last_filtered_probs_normalized = exp_last_log_probs / np.sum(exp_last_log_probs)

        k_with_const = X_train_with_const.shape[1]
        beta_tvtp = initial_params[:k_with_const * n_states].reshape(n_states, k_with_const)
        a_params = initial_params[k_with_const * n_states + n_states:k_with_const * n_states + 2 * n_states]
        b_params = initial_params[k_with_const * n_states + 2 * n_states:]

        # 1-step prediction
        current_X_test_features = X_test.iloc[i:i + 1].values
        z_for_P_matrix = z_test.iloc[i - 1] if i > 0 else (z_train_vals[-1] if len(z_train_vals) > 0 else 0.0)
        pred_1 = predict_tvtp_corrected(
            current_X_test_features,
            z_for_P_matrix,
            last_filtered_probs_normalized,
            beta_tvtp,
            a_params,
            b_params,
            n_states
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
                future_X_test_5_features,
                z_for_P_matrix_5,
                last_filtered_probs_normalized,
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

        # 22-step prediction
        if i + 21 < len(X_test):
            future_X_test_22_features = X_test.iloc[i + 21:i + 22].values
            z_for_P_matrix_22 = z_test.iloc[i + 20] if i + 20 < len(z_test) else (z_test.iloc[-1] if len(z_test) > 0 else 0.0)
            pred_22 = predict_tvtp_corrected(
                future_X_test_22_features,
                z_for_P_matrix_22,
                last_filtered_probs_normalized,
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
        predictions_tvtp5.append(predictions_tvtp5[-1] if predictions_tvtp5 and predictions_tvtp5[-1] is not None else np.nan)
        actuals_tvtp5.append(float(y_test.iloc[i + 4]) if i + 4 < len(y_test) else np.nan)
        predictions_tvtp22.append(predictions_tvtp22[-1] if predictions_tvtp22 and predictions_tvtp22[-1] is not None else np.nan)
        actuals_tvtp22.append(float(y_test.iloc[i + 21]) if i + 21 < len(y_test) else np.nan)

    # Update rolling window to maintain window_size
    new_X = X_test.iloc[i:i + 1].values
    rolling_X = np.vstack((rolling_X[1:], new_X))
    rolling_y = np.append(rolling_y[1:], y_test.iloc[i])
    rolling_z = np.append(rolling_z[1:], z_test.iloc[i])

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

df_predictions_tvtp.to_csv('tvtphar-rs_corrected_DASH_window1000_test500.csv', index=False)
print("结果已保存到 'tvtphar-rs_corrected_DASH_window1000_test500.csv'")
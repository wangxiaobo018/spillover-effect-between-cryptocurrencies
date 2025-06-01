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
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv



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
print(RV)
# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])

# 正确计算滞后特征
rv_lag1 = RV['RV'].shift(1)   # 滞后1期
rv_lag5 = RV['RV'].rolling(window=5).mean().shift(1)
rv_lag22 = RV['RV'].rolling(window=22).mean().shift(1)

data_rv = pd.DataFrame({
    'RV': RV['RV'],
    'rv_lag1': rv_lag1,
    'rv_lag5': rv_lag5,
    'rv_lag22': rv_lag22
})

data_rv = data_rv.dropna()
print(data_rv)

test_size = 300
# 划分训练集和测试集
train_data = data_rv.iloc[:len(data_rv) - test_size]
test_data = data_rv.iloc[len(data_rv) - test_size:]


# 分割特征和目标值
X_train = train_data.drop('RV', axis=1)
y_train = train_data['RV']
X_test = test_data.drop('RV', axis=1)
y_test = test_data['RV']

# 初始化预测和实际值列表
predictions_lr1 = []
actuals_lr1 = []
predictions_lr5 = []
actuals_lr5 = []
predictions_lr22 = []
actuals_lr22 = []

# 初始化滚动窗口
rolling_X = X_train.copy()
rolling_y = y_train.copy()


# 存储第一个窗口的模型统计信息
first_window_stats = {}
# 滚动窗口预测主循环
# 创建一个列表来存储每个窗口的平均值
window_means = []

for i in range(len(X_test)):
    # 训练模型
    if isinstance(rolling_X, np.ndarray):
        X_train = rolling_X
        y_train = rolling_y
    else:
        X_train = rolling_X.values
        y_train = rolling_y.values

    # 使用 statsmodels 进行线性回归以获取详细统计信息（仅第一次）
    if i == 0:
        X_train_sm = sm.add_constant(X_train)  # 添加常数项
        model_sm = sm.OLS(y_train, X_train_sm).fit()

        # 保存参数估计值和标准误
        first_window_stats['params'] = model_sm.params
        first_window_stats['std_err'] = model_sm.bse
        first_window_stats['log_likelihood'] = model_sm.llf
        first_window_stats['aic'] = model_sm.aic
        first_window_stats['bic'] = model_sm.bic

        # 打印出来查看
        print("== First Rolling Window Model Statistics ==")
        print(model_sm.summary())

        # 将系数用于后续预测
        model = LinearRegression()
        model.fit(rolling_X, rolling_y)
    else:
        model = LinearRegression()
        model.fit(rolling_X, rolling_y)

    # 计算当前窗口实际值的均值（无论预测值是否为负）
    window_mean = np.mean(rolling_y)
    window_means.append(window_mean)

    # 1步预测
    pred_1 = model.predict(X_test.iloc[i:i + 1])[0]
    predictions_lr1.append(pred_1)
    actuals_lr1.append(y_test.iloc[i])

    # 5步预测
    if i + 4 < len(X_test):
        pred_5 = model.predict(X_test.iloc[i + 4:i + 5])[0]
        predictions_lr5.append(pred_5)
        actuals_lr5.append(y_test.iloc[i + 4])
    else:
        predictions_lr5.append(None)
        actuals_lr5.append(None)

    # 22步预测
    if i + 21 < len(X_test):
        pred_22 = model.predict(X_test.iloc[i + 21:i + 22])[0]
        predictions_lr22.append(pred_22)
        actuals_lr22.append(y_test.iloc[i + 21])
    else:
        predictions_lr22.append(None)
        actuals_lr22.append(None)

    # 更新滚动窗口
    if isinstance(rolling_X, np.ndarray):
        new_obs = X_test.iloc[i:i + 1].values
        rolling_X = np.vstack((rolling_X[1:], new_obs))
        rolling_y = np.append(rolling_y[1:], y_test.iloc[i])
    else:
        rolling_X = pd.concat([rolling_X.iloc[1:], X_test.iloc[i:i + 1]])
        rolling_y = pd.concat([rolling_y.iloc[1:], pd.Series([y_test.iloc[i]])])

# 创建结果DataFrame
df_predictions_lr = pd.DataFrame({
    'Prediction_1': predictions_lr1,
    'Actual_1': actuals_lr1,
    'Prediction_5': predictions_lr5,
    'Actual_5': actuals_lr5,
    'Prediction_22': predictions_lr22,
    'Actual_22': actuals_lr22,
    'Window_Mean': window_means  # 添 加每个窗口的平均值
})

df_predictions_lr.to_csv('har-rv.csv', index=False)


predictions = df_predictions_lr['Prediction_1'].values
actuals = df_predictions_lr['Actual_1'].values
mse = np.mean((predictions - actuals) ** 2)

# MAE
mae = np.mean(np.abs(predictions - actuals))

# HMSE
hmse = np.mean((1 - predictions / actuals) ** 2)

# HMAE
hmae = np.mean(np.abs(1 - predictions / actuals))


qlike = np.mean(np.log(predictions) + actuals / predictions)

# 打印结果
print(f"1-Step Prediction Loss Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"HMSE: {hmse:.6f}")
print(f"HMAE: {hmae:.6f}")
print(f"QLIKE: {qlike:.6f}")

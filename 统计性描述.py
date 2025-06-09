
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from concurrent.futures import ThreadPoolExecutor
import warnings

from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
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


# Read the data
'''
# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")



data_filtered = df[df['code'] == "BTC"].copy()


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

        # 计算RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())


        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))


        term1 = abs_ret.iloc[4:].values  # Ret[5:n()]
        term2 = abs_ret.iloc[2:-2].values  # Ret[3:(n-2)]
        term3 = abs_ret.iloc[:-4].values  # Ret[1:(n-4)]

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

        # calculate JV
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({

            'BV': BV,
            'JV': JV,
            'C_t': C_t
        })


    result_df = pd.DataFrame(results)
    return result_df[['BV', 'JV', 'C_t']]

har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)
print(har_cj)


# Read the data

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
    returns = (prices / prices.shift(1) - 1) * 100  # (Pt - Pt-1) / Pt-1 * 100
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

# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])


data_get_cj = pd.merge(RV, har_cj, left_index=True, right_index=True)

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

# Assuming data_get_cj is already defined
# print(data_get_cj)
selected_data = data_get_cj[['RV', 'JV', 'C_t']]

stats = {}
for column in selected_data.columns:
    series = selected_data[column].dropna()  # 去除缺失值
    stats[column] = {
        'Mean': np.mean(series),
        'Std Dev': np.std(series),
        'Skewness': skew(series),
        'Kurtosis': kurtosis(series),
        'Max': np.max(series),
        'Min': np.min(series),
        'ADF': adfuller(series)[0],  # 返回ADF统计量
    }

    # 使用L-B(10)检验 - 修正版本
    lb_test = acorr_ljungbox(series, lags=10, return_df=True)
    stats[column]['L-B(10)'] = lb_test['lb_stat'].iloc[-1]  # 获取lag=10的统计量

    # 或者使用return_df=False的版本：
    # lb_test = acorr_ljungbox(series, lags=10, return_df=False)
    # stats[column]['L-B(10)'] = lb_test.statistic.iloc[-1]  # 获取最后一个lag的统计量

# 转换为DataFrame以便查看
stats_df = pd.DataFrame(stats).T

# 显示所有列
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽

# 打印结果
print(stats_df)
print("\nLaTeX format:")
print(stats_df.to_latex())


from pathlib import Path
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

def process_har_rs_model(data_idx_path):
    """Process data for HAR-RS model."""
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
    data_cj = data_ret_idx.query('id == "BTC"').copy()

    # Calculate RS statistics by date
    result = (
        data_cj.groupby(pd.to_datetime(data_cj['DT']).dt.date)
        .apply(calculate_RS)
        .reset_index()
    )

    # Rename columns for consistency
    result = result.rename(columns={'level_0': 'DT'})

    return result

# Usage example:
if __name__ == "__main__":
    data_idx_path = Path("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")

    final_data = process_har_rs_model(data_idx_path)
    print("\nFinal Data Sample:")
    print(final_data.head())


# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")

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
data_filtered = data_ret[data_ret['id'] == "BTC"].copy()
data_filtered = data_filtered.drop('id', axis=1)

# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure both 'DT' columns are datetime
RV['DT'] = pd.to_datetime(RV['DT'])

final_data['DT'] = pd.to_datetime(final_data['DT'])

# Merge the dataframes
data_rs = pd.merge(RV, final_data, on='DT', how='inner')

# Display the merged dataframe
print(data_rs.head())



# Assuming data_get_cj is already defined
# print(data_get_cj)
selected_data = data_rs[['RS_plus', 'RS_minus']]

stats = {}
for column in selected_data.columns:
    series = selected_data[column].dropna()  # 去除缺失值
    stats[column] = {
        'Mean': np.mean(series),
        'Std Dev': np.std(series),
        'Skewness': skew(series),
        'Kurtosis': kurtosis(series),
        'Max': np.max(series),
        'Min': np.min(series),
        'ADF': adfuller(series)[0],  # 返回ADF统计量
    }

    # 使用L-B(10)检验 - 修正版本
    lb_test = acorr_ljungbox(series, lags=10, return_df=True)
    stats[column]['L-B(10)'] = lb_test['lb_stat'].iloc[-1]  # 获取lag=10的统计量

    # 或者使用return_df=False的版本：
    # lb_test = acorr_ljungbox(series, lags=10, return_df=False)
    # stats[column]['L-B(10)'] = lb_test.statistic.iloc[-1]  # 获取最后一个lag的统计量

# 转换为DataFrame以便查看
stats_df = pd.DataFrame(stats).T

# 显示所有列
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽

# 打印结果
print(stats_df)
print("\nLaTeX format:")
print(stats_df.to_latex())
'''


# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/btc.csv")

data_filtered = df[df['code'] == "BTC"].copy()
# 按组进行分类统计
group_summary = df.groupby('code').size().reset_index(name='NumObservations')

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

# 使用函数
har_re = get_re(data_filtered, alpha=0.05)
print(har_re)


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
data_filtered = data_ret[data_ret['id'] == "BTC"].copy()
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


# Ensure 'har_cj' is properly prepared before merging
har_pd_re = pd.merge(RV, har_re, left_index=True, right_index=True)
print(har_pd_re)

selected_data = har_pd_re[['REX_minus', 'REX_plus', 'REX_moderate']]

stats = {}
for column in selected_data.columns:
    series = selected_data[column].dropna()  # 去除缺失值
    stats[column] = {
        'Mean': np.mean(series),
        'Std Dev': np.std(series),
        'Skewness': skew(series),
        'Kurtosis': kurtosis(series),
        'Max': np.max(series),
        'Min': np.min(series),
        'ADF': adfuller(series)[0],  # 返回ADF统计量
    }

    # 使用L-B(10)检验 - 修正版本
    lb_test = acorr_ljungbox(series, lags=10, return_df=True)
    stats[column]['L-B(10)'] = lb_test['lb_stat'].iloc[-1]  # 获取lag=10的统计量

    # 或者使用return_df=False的版本：
    # lb_test = acorr_ljungbox(series, lags=10, return_df=False)
    # stats[column]['L-B(10)'] = lb_test.statistic.iloc[-1]  # 获取最后一个lag的统计量

# 转换为DataFrame以便查看
stats_df = pd.DataFrame(stats).T

# 显示所有列
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽

# 打印结果
print(stats_df)
print("\nLaTeX format:")
print(stats_df.to_latex())
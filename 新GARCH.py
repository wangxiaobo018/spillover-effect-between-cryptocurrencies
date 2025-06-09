import pandas as pd
import numpy as np
from datetime import date
from arch import arch_model
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- 1. 加载和预处理数据 (与之前相同) ---
try:
    df_high_freq = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv")
except FileNotFoundError:
    print("错误：高频数据文件未找到。请检查文件路径是否正确。")
    exit()

df_processed = df_high_freq[['time', 'code', 'close']].copy()
df_processed.columns = ['DT', 'id', 'PRICE']
df_processed['DT'] = pd.to_datetime(df_processed['DT'], errors='coerce')
df_processed = df_processed.dropna(subset=['DT', 'PRICE'])
df_btc = df_processed[df_processed['id'] == "BTC"].copy()
df_btc = df_btc.sort_values(by='DT')
daily_prices = df_btc.set_index('DT')['PRICE'].resample('D').last().ffill()
daily_data = daily_prices.reset_index()
daily_data.columns = ['DT', 'Daily_Close']
daily_data['Daily_Return'] = daily_data['Daily_Close'].pct_change().fillna(0) * 100
daily_data = daily_data.dropna()

start_date = date(2019, 3, 28)
end_date = date(2025, 3, 30)
daily_data_filtered = daily_data[
    (daily_data['DT'].dt.date >= start_date) &
    (daily_data['DT'].dt.date <= end_date)
    ].copy()
daily_returns_series = daily_data_filtered.set_index('DT')['Daily_Return']

print(f"--- 最终准备好的GARCH模型输入 (从 {start_date} 到 {end_date}) ---")
print(f"数据点数量: {len(daily_returns_series)}")
print(daily_returns_series.head())

# --- 2. GARCH模型滚动预测 ---
models_to_test = {
    'GARCH': {'vol': 'Garch', 'p': 1, 'o': 0, 'q': 1},
    'EGARCH': {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1},
    'GJR_GARCH': {'vol': 'Garch', 'p': 1, 'o': 1, 'q': 1},  # GJR-GARCH is an alias for GARCH with o>0
    'APARCH': {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1},
    'GARCH-M': {'vol': 'Garch', 'p': 1, 'o': 0, 'q': 1, 'power': 2.0, 'mean': 'ARX'},
    'FIGARCH': {'vol': 'FIGARCH', 'p': 1, 'q': 1}
}

horizons = [1, 5, 22]
test_size = 500
window_size = 1000

if window_size <= 0:
    print(f"错误：数据量 ({len(daily_returns_series)}) 不足以支持窗口大小。")
    exit()

print(f"\n滚动窗口大小: {window_size}")
print(f"将要进行的预测步数 (测试集大小): {test_size}")
print(f"预测 horizons: {horizons} 天")

# 步骤1: 先生成并存储方差 σ²
column_names = [f"{model_name}_h{h}" for model_name in models_to_test.keys() for h in horizons]
variance_predictions_df = pd.DataFrame(
    index=daily_returns_series.index[window_size:],
    columns=column_names,
    dtype=float
)

# --- 滚动预测主循环 ---
for model_name, model_params in models_to_test.items():
    print(f"\n--- 正在进行 {model_name} 的多步滚动预测 (预测目标: σ²)... ---")
    for i in tqdm(range(test_size)):
        train_data = daily_returns_series.iloc[i: i + window_size]
        am = arch_model(train_data, **{k: v for k, v in model_params.items() if
                                       k != 'vol' and k != 'p' and k != 'o' and k != 'q' and k != 'mean'},
                        vol=model_params['vol'], p=model_params.get('p', 1), o=model_params.get('o', 0),
                        q=model_params.get('q', 1), mean=model_params.get('mean', 'Constant'))
        try:
            res = am.fit(update_freq=0, disp='off')
            # 使用模拟方法处理像FIGARCH这类模型的多步预测
            temp_forecast = res.forecast(horizon=max(horizons), method='simulation', reindex=False)
            predicted_variances_series = temp_forecast.variance.iloc[0]

            current_index = daily_returns_series.index[window_size + i]
            for h in horizons:
                column_name = f"{model_name}_h{h}"
                variance_predictions_df.loc[current_index, column_name] = predicted_variances_series.iloc[h - 1]
        except Exception as e:
            # print(f"模型 {model_name} 在索引 {i} 处拟合失败: {e}")
            pass  # 如果某个点拟合失败，暂时跳过

print("\n--- 多步滚动预测完成 ---")
print("预测的条件方差 σ² (前5条):")
print(variance_predictions_df.head())

# --- 3. 结果处理、转换与重构 (核心修正点) ---
print("\n--- 正在处理最终结果... ---")

# 步骤2: 将所有方差预测值进行开方，得到波动率 σ
print("1. 将方差 σ² 转换为波动率 σ (开方)...")
volatility_predictions_df = np.sqrt(variance_predictions_df)
print("波动率 σ 预测 (前5条):")
print(volatility_predictions_df.head())

# 步骤3: 按预测步长h进行列重构
print("\n2. 重构DataFrame，按预测步长 h 分组...")
all_dfs_by_horizon = []
for h in horizons:
    cols_h = [f"{model_name}_h{h}" for model_name in models_to_test.keys()]
    df_h = volatility_predictions_df[cols_h].copy()

    # 为了清晰，可以重命名列，去掉 "_h{h}" 后缀
    df_h.columns = [model_name for model_name in models_to_test.keys()]

    # 创建一个多重索引，指明这是哪个horizon的预测
    df_h.columns = pd.MultiIndex.from_product([[f'h={h}'], df_h.columns],
                                              names=['horizon', 'model'])
    all_dfs_by_horizon.append(df_h)

# 将所有按步长分组的DataFrame横向拼接
final_structured_df = pd.concat(all_dfs_by_horizon, axis=1)

print("\n重构后的DataFrame结构 (前5条):")
print(final_structured_df.head())

# --- 4. 添加实际波动率代理并保存 ---

# 对于GARCH的波动率(σ)，最常用的实际值代理是日收益率的绝对值 |r|
# (对于方差σ²，代理是 r²)
actual_volatility_proxy = pd.DataFrame(
    np.abs(daily_returns_series.iloc[window_size:]),
    index=final_structured_df.index
)
actual_volatility_proxy.columns = ['Actual_Volatility_Proxy']

# 将实际值代理加到最终结果的最前面
final_output_df = pd.concat([actual_volatility_proxy, final_structured_df], axis=1)

# 保存为CSV，多重索引会自动处理
final_output_df.to_csv("garch_family_multi_horizon_structured_forecasts.csv")

print("\n--- 任务完成 ---")
print("结构化、已开方的多步预测结果已保存到 garch_family_windows.csv")
print("文件头部预览:")
print(final_output_df.head())
#!/usr/bin/env python3.11
import yfinance as yf
import pandas as pd
import numpy as np

# Get data for 2024-2025 to understand what happened
print("=== ANALYZING WHY HELIX 1.1 FAILED IN 2024-2025 ===\n")

spy_data = yf.download('SPY', start='2024-01-01', end='2025-08-31', auto_adjust=True)['Close']
print('SPY 2024-2025 Performance:')
print('Start: ${:.2f}'.format(float(spy_data.iloc[0])))
print('End: ${:.2f}'.format(float(spy_data.iloc[-1])))
spy_return = (spy_data.iloc[-1] / spy_data.iloc[0]) - 1
print('Total Return: {:.2%}'.format(float(spy_return)))

# Get our factor ETFs
etfs = ['SPY', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
data = yf.download(etfs, start='2024-01-01', end='2025-08-31', auto_adjust=True)['Close']

print('\n=== Factor ETF Performance 2024-2025 ===')
for etf in etfs:
    if etf in data.columns:
        start_price = float(data[etf].iloc[0])
        end_price = float(data[etf].iloc[-1])
        ret = (end_price / start_price) - 1
        print('{}: {:.2%} (${:.2f} -> ${:.2f})'.format(etf, ret, start_price, end_price))

# Check if we have equal weights what would have happened
print('\n=== EQUAL WEIGHT BENCHMARK ===')
equal_weight_returns = []
for etf in etfs:
    if etf in data.columns:
        ret = (data[etf].iloc[-1] / data[etf].iloc[0]) - 1
        equal_weight_returns.append(float(ret))

equal_weight_performance = np.mean(equal_weight_returns)
print('Equal weight portfolio return: {:.2%}'.format(equal_weight_performance))
print('vs Helix 1.1: 10.20%')
print('vs SPY: {:.2%}'.format(float(spy_return)))

# Let's check monthly returns to see the pattern
print('\n=== MONTHLY PERFORMANCE BREAKDOWN ===')
spy_monthly = spy_data.resample('M').last().pct_change().dropna()
print('SPY Monthly Returns 2024-2025:')
for date, ret in spy_monthly.items():
    print('{}: {:.2%}'.format(date.strftime('%Y-%m'), float(ret)))

# Check what our strategy would have allocated to
print('\n=== LIKELY ISSUE: LOW VOLATILITY BIAS ===')
usmv_return = (data['USMV'].iloc[-1] / data['USMV'].iloc[0]) - 1
print('USMV (Low Vol) return: {:.2%}'.format(float(usmv_return)))
print('This is probably where our strategy allocated most weight due to "regime persistence"')
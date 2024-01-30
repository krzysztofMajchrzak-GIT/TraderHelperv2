import json
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
from operator import itemgetter
from scipy.signal import savgol_filter


def funkcja_obciągająca(interval, window_size, endTime=None):
    df_columns = ['unix', 'time', 'open', 'high', 'low', 'close', 'Volume ETH']
    url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval={'1m'}&limit={window_size}"
    if endTime:
        endTime = endTime.tz_localize(timezone('UTC'))
        url += f"&endTime={int(endTime.timestamp() * 1000)}"
    content = []
    while len(content) < window_size:
        try:
            if endTime is None and len(content):
                r = requests.get(url + f"&endTime={content[0][0]}")
            else:
                r = requests.get(url)
            content.extend(json.loads(r.content))
        except Exception as ex:
            print(ex)
            continue
    df = pd.DataFrame.from_records(content, columns=['unix', 'open', 'high', 'low', 'close', 'Volume ETH', 'close_timestamp',
                                   'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['time'] = df.unix.apply(lambda ts: datetime.fromtimestamp(ts/1000, timezone("Europe/Warsaw")))
    df = df[df_columns].sort_values('time')
    for column in df_columns:
        if column == 'time' or column == 'unix':
            continue
        df[column] = df[column].astype(float)
    return df


def process(df, interval, modifier, columns):
    COLUMN_SET = set(columns)
    window_size = 10

    # Set index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Resample the DataFrame to specified period and use the last value within each
    df = df[['open', 'high', 'low', 'close', 'Volume ETH', 'unix']].resample(interval).agg(
        {'open': 'first', 'high': np.max, 'low': np.min, 'close': 'last', 'Volume ETH': np.sum, 'unix': 'first'})

    # Apply the Savitzky-Golay filter to all columns
    df = df.apply(lambda col: savgol_filter(col, 10, 2))

    # Reset index
    df = df.reset_index(drop=False)

    # Calculate the difference between each value and its previous value
    df['Change'] = df['close'].diff()

    if modifier < 0:
        # Add a column for the trend label (1: uptrend, 0: downtrend)
        df['Trend'] = df['Change'].apply(lambda x: 1 if x > 0 else 0)

        # Create a new column B with the logical negation of column A
        df['-Trend'] = (~df['Trend'].astype(bool)).astype(int)

    else:
        last_index = df.shape[0] - 1

        df['sign'] = df['close'].diff().replace(0, method='ffill').fillna(0) * df['close'].diff(-1).replace(0, method='ffill').fillna(0)
        df['ext_index'] = (pd.Series(range(df.shape[0])) * (df['sign'] > 0).astype(int)).replace(0, method='bfill')
        df['ext_index'] = df['ext_index'].replace(0, last_index).shift(-1, fill_value=last_index)
        df['ext_values'] = df['close'].loc[df['ext_index']].reset_index(drop=True)
        df['ext_diff'] = df['close'] - df['ext_values']

        df['short'] = df['ext_diff'].apply(lambda x: 1 if x > 0 else 0)
        df['long'] = (~df['short'].astype(bool)).astype(int)

        if modifier > 0:
            df['null'] = df['close'] * modifier > df['ext_diff'].abs()
            df['-null'] = (~df['null'].astype(bool)).astype(int)
            df['null'] = df['null'].astype(int)
            # Set nan on null in order to shift previous label forward
            df['-null'] = df['-null'].replace(0, np.nan)
            i = (df['null'] == 0).idxmax()
            short = df['short'][:i]
            long = df['long'][:i]
            i -= 1
            df['short'] *= df['-null']
            df['long'] *= df['-null']
            df.loc[:i, 'short'] = short
            df.loc[:i, 'long'] = long
            df['short'] = df['short'].fillna(method='ffill').astype(int)
            df['long'] = df['long'].fillna(method='ffill').astype(int)

    if 'Coppock' in COLUMN_SET:
        # Calculate the 14-period Rate of Change (ROC) using Close prices
        df['ROC'] = df['close'].pct_change(14) * 100

        # Calculate the 11-period Rate of Change (ROC) using the 14-period ROC
        df['ROC_Signal'] = df['close'].pct_change(11) * 100

        # Calculate the 10-period Weighted Moving Average (WMA) of the 11-period ROC
        df['WMA'] = df['ROC_Signal'].rolling(window=10).apply(lambda x: np.dot(x, np.arange(1, 11)) / 55, raw=True)

        # Calculate the 14-period Weighted Moving Average (WMA) of the 10-period WMA
        df['Coppock'] = df['WMA'].rolling(window=14).apply(lambda x: np.dot(x, np.arange(1, 15)) / 105, raw=True)

    if 'RSI' in COLUMN_SET or 'StochRSI' in COLUMN_SET:
        # Calculate gains (positive changes) and losses (negative changes), and the average gain and average loss over a specified period
        df['AvgGain'] = df['Change'].apply(lambda x: x if x > 0 else 0).rolling(window=window_size).mean()
        df['AvgLoss'] = df['Change'].apply(lambda x: abs(x) if x < 0 else 0).rolling(window=window_size).mean()

        # Calculate the relative strength (RS)
        df['RS'] = df['AvgGain'] / df['AvgLoss']

        # Calculate the RSI
        df['RSI'] = 100 - (100 / (1 + df['RS']))

    if 'StochRSI' in COLUMN_SET:
        # Normalize the RSI values to a range between 0 and 1
        min_rsi = df['RSI'].rolling(window=window_size).min()
        max_rsi = df['RSI'].rolling(window=window_size).max()
        df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)

    if 'ROC' in COLUMN_SET:
        # Calculate the 10-period Rate of Change (ROC) using Close prices
        df['ROC'] = df['close'].pct_change(10) * 100

    if 'MACD' in COLUMN_SET:
        # Define the parameters
        short_ema_period = 12
        long_ema_period = 26
        signal_ema_period = 9

        # Calculate the short-term EMA
        df['EMA_short'] = df['close'].ewm(span=short_ema_period, adjust=False).mean()

        # Calculate the long-term EMA
        df['EMA_long'] = df['close'].ewm(span=long_ema_period, adjust=False).mean()

        # Calculate the MACD Line
        df['MACD_Line'] = df['EMA_short'] - df['EMA_long']

        # Calculate the Signal Line
        df['Signal_Line'] = df['MACD_Line'].ewm(span=signal_ema_period, adjust=False).mean()

        # Calculate the MACD Histogram
        df['MACD'] = df['MACD_Line'] - df['Signal_Line']

    if 'Senkou_span_a' in COLUMN_SET or 'Sen' in COLUMN_SET or 'Tenkan_sen' in COLUMN_SET:
        # Define the parameters for Ichimoku Cloud
        conversion_period = 9
        base_period = 26
        lagging_period = 26
        leading_period = 52

        # Calculate the Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(window=conversion_period).max()
        low_9 = df['low'].rolling(window=conversion_period).min()
        df['Tenkan_sen'] = ((high_9 + low_9) / 2).fillna(method='bfill').replace(0, method='bfill')

    if 'Senkou_span_a' in COLUMN_SET or 'Sen' in COLUMN_SET or 'Kijun_sen' in COLUMN_SET:
        # Calculate the Kijun-sen (Base Line)
        high_26 = df['high'].rolling(window=base_period).max()
        low_26 = df['low'].rolling(window=base_period).min()
        df['Kijun_sen'] = ((high_26 + low_26) / 2).fillna(method='bfill').replace(0, method='bfill')

    if 'Sen' in COLUMN_SET:
        # Calculate the difference - the meaningful indicator
        df['Sen'] = df['Tenkan_sen'] - df['Kijun_sen']

    if 'Senkou_span_a' in COLUMN_SET:
        # Calculate the Senkou Span A (Leading Span A)
        df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(lagging_period).fillna(method='bfill').replace(0, method='bfill')

    if 'Senkou_span_b' in COLUMN_SET:
        # Calculate the Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(window=leading_period).max()
        low_52 = df['low'].rolling(window=leading_period).min()
        df['Senkou_span_b'] = ((high_52 + low_52) / 2).shift(lagging_period).fillna(method='bfill').replace(0, method='bfill')

    if 'Kumo' in COLUMN_SET:
        # Calculate the difference - the meaningful indicator
        df['Kumo'] = df['Senkou_span_a'] - df['Senkou_span_b']

    if 'Chikou_span' in COLUMN_SET:
        # Calculate the Chikou Span (Lagging Span)
        df['Chikou_span'] = df['close'].shift(-lagging_period).fillna(method='ffill').replace(0, method='ffill')

    if 'Aroon_Oscillator' in COLUMN_SET:
        # Define the number of periods for the Aroon Indicator
        period = 14

        # Calculate the number of periods since the highest high
        df['Periods_Since_Highest'] = df['high'].rolling(window=period).apply(lambda x: period - x.argmax(), raw=True).shift(1).fillna(method='bfill')

        # Calculate the number of periods since the lowest low
        df['Periods_Since_Lowest'] = df['low'].rolling(window=period).apply(lambda x: period - x.argmin(), raw=True).shift(1).fillna(method='bfill')

        # Calculate the Aroon Up and Aroon Down values
        df['Aroon_Up'] = (period - df['Periods_Since_Highest']) / period * 100
        df['Aroon_Down'] = (period - df['Periods_Since_Lowest']) / period * 100

        # Calculate the Aroon Oscillator
        df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']

    # Define the Fibonacci retracement ratios
    fibonacci_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

    # Calculate Fibonacci Retracement levels for each row in the DataFrame
    for ratio in fibonacci_ratios:
        label = f'Fibonacci_{int(ratio * 100)}'
        if label in COLUMN_SET:
            df[label] = (df['high'] - df['low']) * ratio + df['low']

    if 'OBV' in COLUMN_SET:
        # Calculate the sign of the price change
        price_sign = pd.Series(0, index=df.index)
        price_sign[df['Change'] > 0] = 1
        price_sign[df['Change'] < 0] = -1

        # Calculate the OBV using cumulative sum based on the price sign and volume
        df['OBV'] = (price_sign * df['Volume ETH']).cumsum()

    if 'Price_strength' in COLUMN_SET:
        df['Price_strength'] = (df['close'] - df['low']) / (df['high'] - df['low']) * 100  # Price_strength index, 100 if close at highs, 0 if close at lows
    if 'Relative_volume' in COLUMN_SET:
        df['Relative_volume'] = round(df['Volume ETH'] / df['Volume ETH'].rolling(window=20).mean(), 3)  # Measure of volume strength
    if 'Close_Open' in COLUMN_SET:
        df['Close_Open'] = df['close'] / df['open']  # >1 if close higher than open, <1 if otherwise

    if 'Hurst' in COLUMN_SET:
        max_length = len(df['close']) // 2 + 1
        chunk_count = 1
        r = []
        while chunk_count < max_length:
            r.append((np.ceil(len(df['close']) // chunk_count), np.mean([np.ptp(np.cumsum(chunk - np.mean(chunk))) / np.std(chunk)
                                                                         for chunk in np.array_split(df['close'], chunk_count)])))
            chunk_count *= 2
        df['Hurst'] = np.polyfit(*np.transpose(np.log(r)), 1)[0]

        # Generate a range of lags to be used for rescaled range analysis
        lags = range(2, len(df['close']) // 2)

        # Calculate the standard deviation of the differences between elements at different lags
        tau = [np.std(np.subtract(df['close'][lag:], df['close'][:-lag])) for lag in lags]

        # Fit a linear regression model to the log-log plot of R/S against the time window size (lags)
        df['Hurst'] = np.polyfit(np.log(lags), np.log(tau), 1)[0]

    # Replace NaN values with zero
    df = df.fillna(0)

    # Set index again
    df.set_index('time', inplace=True)

    df['unix'] = df['unix'].astype(int)
    return df[columns + ['close'] + (['long', 'short', 'unix'] if modifier >= 0 else ['Trend', '-Trend'])]


def get_file(df, *args):
    data = df[args]
    data['label'] = df.apply(lambda row: 'null' if row['null'] else 'short' if row['short'] else 'long' if row['long'] else 'error', axis=1)
    data = data.reset_index()
    data['time'] = data['time'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
    data.to_csv('out.csv')


def normalize(df, columns, path):
    """Normalize columns between -1 and 1"""
    if os.path.exists(path):
        with open(path) as f:
            params = json.loads(f.read())
    else:
        params = None
        new_params = {}
    for i in filter(lambda j: j not in {'Trend', '-Trend', 'long', 'short'}, columns):
        if params:
            min, max = itemgetter('min', 'max')(params[i])
        else:
            min = df[i].min()
            max = df[i].max()
            new_params[i] = {
                'min': min,
                'max': max
            }
        df[i] = (df[i] - min) / (max - min) * 2 - 1
    if params is None:
        with open(path, 'w') as f:
            f.write(json.dumps(new_params))
    return df

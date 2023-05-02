import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time 

# connect to the MetaTrader 5 terminal
mt5.initialize()

# define the forex pair to use
symbol = 'USDJPY.HKT'

# define the timeframe
timeframe = mt5.TIMEFRAME_M15

# define the hyperparameters for the model
conversion_line_period = 9
base_line_period = 26
leading_span_b_period = 52
displacement = 26
n_estimators = 100

# get the historical data from MT5
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10000)

# create a DataFrame from the historical data
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# calculate the Ichimoku indicators
df['conversion_line'] = (df['high'].rolling(window=conversion_line_period).max() + df['low'].rolling(window=conversion_line_period).min()) / 2
df['base_line'] = (df['high'].rolling(window=base_line_period).max() + df['low'].rolling(window=base_line_period).min()) / 2
df['leading_span_a'] = (df['conversion_line'] + df['base_line']) / 2
df['leading_span_b'] = (df['high'].rolling(window=leading_span_b_period).max() + df['low'].rolling(window=leading_span_b_period).min()) / 2
df['leading_span_a'] = df['leading_span_a'].shift(displacement)
df['leading_span_b'] = df['leading_span_b'].shift(displacement)

# create the trading signals
df['signal'] = 0
df.loc[df['conversion_line'] > df['base_line'], 'signal'] = 1
df.loc[df['conversion_line'] < df['base_line'], 'signal'] = -1
df['signal'] = df['signal'].shift(1)

# calculate the returns
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['signal'] * df['returns']

# split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]
train_df = train_df.dropna(subset=['signal'])
train_df = train_df.fillna(train_df.mean())

# impute missing values in the training set
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = train_df[['conversion_line', 'base_line', 'leading_span_a', 'leading_span_b']]
X_train = imputer.fit_transform(X_train)
y_train = train_df['signal'].values

# train the model
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# create a DataFrame to store the results
results = pd.DataFrame(columns=['datetime', 'action', 'lots', 'price', 'profit'])

# define the lot size and stop loss
lot_size = 0.1
stop_loss = 100 # in points
# iterate over the test data and generate trading signals
prev_conversion_price = None
signals = []
for i in range(len(test_df)):
    row = test_df.iloc[i]
    conversion_price = row['conversion_line']
    base_price = row['base_line']

    # calculate the price difference between the current row and the previous row
    if prev_conversion_price is not None:
        price_diff = conversion_price - prev_conversion_price

        # determine whether to buy, sell or hold
        if price_diff > 0:
            signal = 'buy'
        elif price_diff < 0:
            signal = 'sell'
        else:
            signal = 'hold'
            
        # append the trading signal to the signals list
        signals.append(signal)

    # set the previous conversion price to the current conversion price
    prev_conversion_price = conversion_price

# append the signals list to the test data
signals = np.append(signals, np.nan)
test_df['Signals'] = signals

# print the test data with signals
print(test_df.head())

test_df.to_excel('results.xlsx', index=False)

    


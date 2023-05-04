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
symbols = ['USDJPY.HKT', 'EURUSD.HKT', 'CADJPY.HKT']

test = []

for symbol in symbols:
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
    model = RandomForestClassifier(n_estimators = n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # evaluate the model on the testing set
    X_test = test_df[['conversion_line', 'base_line', 'leading_span_a', 'leading_span_b']]
    X_test = imputer.transform(X_test)
    y_test = test_df['signal'].values
    y_pred = model.predict(X_test)
    test = test_df


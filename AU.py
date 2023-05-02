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
symbol = 'AUDUSD.HKT'

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

# evaluate the model on the testing set
X_test = test_df[['conversion_line', 'base_line', 'leading_span_a', 'leading_span_b']]
X_test = imputer.transform(X_test)
y_test = test_df['signal'].values
y_pred = model.predict(X_test)


def close_position(order_type, ticket):
    """Close an open position with the specified ticket number."""
    # check if the position exists
    position = mt5.positions_get(ticket=ticket)
    if len(position) == 0:
        print(f"No position found with ticket number {ticket}")
        return
    # close the position
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    ask = symbol_info.ask
    bid = symbol_info.bid
    if ask == 0 or bid == 0:
        print("Ask or bid price is zero")
        return
    
    # set the price depending on the order type
    if order_type == mt5.ORDER_TYPE_BUY:
        price = ask
    elif order_type == mt5.ORDER_TYPE_SELL:
        price = bid
    else:
        print("Invalid order type")
        return
    request = {
        "action": mt5.TRADE_ACTION_CLOSE_BY,
        "symbol": symbol,
        "volume": 0.01,
        "type": order_type,
        "position": ticket,
        "price": price,
        "magic": 234000,
        "comment": "python market close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    print(f"Position closed with {result}")
    

def open_position(symbol, order_type, lot):
    """Open a market order position with the specified symbol, type and lot size"""
    if not mt5.initialize():
        print("initialize() failed")
        return
    
    # get the current ask and bid prices
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"{symbol} not found, can not call order_check()")
        return
    
    point = symbol_info.point
    ask = symbol_info.ask
    bid = symbol_info.bid
    if ask == 0 or bid == 0:
        print("Ask or bid price is zero")
        return
    
    # set the price depending on the order type
    if order_type == mt5.ORDER_TYPE_BUY:
        price = ask
    elif order_type == mt5.ORDER_TYPE_SELL:
        price = bid
    else:
        print("Invalid order type")
        return
    
    # create a request for a market order
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "magic": 234000,
        "comment": "python market order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # send the request and check the result
    result = mt5.order_send(request)
      
    print(f"Position opened with {result}")

while True:
    # get the latest data from MT5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10000)

    # create a DataFrame from the rates
    df = pd.DataFrame(rates)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # resample the DataFrame to the desired timeframe
    df.set_index('time', inplace=True)
    df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'})
    df.dropna(inplace=True)

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
    X = df[['conversion_line', 'base_line', 'leading_span_a', 'leading_span_b']].values
    y = np.where(df['signal'].values > 0, 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # impute missing values in the training set
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = train_df[['conversion_line', 'base_line', 'leading_span_a', 'leading_span_b']]
    X_train = imputer.fit_transform(X_train)
    y_train = train_df['signal'].values
    
    # train the model on the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)

    # evaluate the model on the testing set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # continuously trade the symbol
    positions = mt5.positions_get(symbol=symbol)
    if len(positions) > 0:
        # close the position if the signal changes to the opposite direction
        if (positions[0].type == mt5.ORDER_TYPE_BUY and y[-1] < 0) or (positions[0].type == mt5.ORDER_TYPE_SELL and y[-1] > 0):
            print(f"Closing {positions[0].type} position {positions[0].ticket} at {mt5.symbol_info_tick(symbol).bid}")
            close_position(positions[0].ticket)
        else:
            print(f"Position {positions[0].type} {positions[0].ticket} open at {positions[0].price_open}")
        # open a new position if there is no current position or the signal changes direction
    if len(positions) == 0:
        lot_size = 0.01
        if y[-1] > 0:
            print(f"Opening buy position with {lot_size} lots at {mt5.symbol_info_tick(symbol).ask}")
            open_position(symbol,mt5.ORDER_TYPE_BUY, lot_size)
        elif y[-1] < 0:
            print(f"Opening sell position with {lot_size} lots at {mt5.symbol_info_tick(symbol).bid}")
            open_position(symbol,mt5.ORDER_TYPE_SELL, lot_size)

        # wait for the next tick
    time.sleep(60) # adjust this to your desired timeframe. 5 seconds is used as an example. 

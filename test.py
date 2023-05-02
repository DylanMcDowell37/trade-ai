import MetaTrader5 as mt5
mt5.initialize()
positions = mt5.positions_get(symbol='USDJPY.HKT')
print(positions)
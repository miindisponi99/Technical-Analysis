import warnings

from datetime import date
from ib_insync import MarketOrder, StopOrder, LimitOrder, Contract
from TA_strats import *

warnings.filterwarnings('ignore', category=FutureWarning)

def signals_aaron_x(data, asset, n=35, capital=100):
    data = data.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low"})
    aroon_data = strategy_AROON_Xcross(data, n=n)
    signals = []
    print("AaronX Strategy")
    for i in range(1, len(aroon_data)):
        current_date = aroon_data.index[i]
        price = aroon_data['close'].iloc[i]
        quantity = capital / price
        if aroon_data['LONG'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AaronX. Long Entry BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif aroon_data['EXIT_LONG'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonX. Exit Long SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif aroon_data['SHORT'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonX. Short Entry SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif aroon_data['EXIT_SHORT'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonX. Exit Short BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")

    return signals


def signals_aaron_y(data, asset, n=50, level=50, capital=100):
    data = data.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low"})
    aroon_data = strategy_AROON_Ycross(data, n=n, livello=level)
    signals = []
    print("AaronY Strategy")
    for i in range(1, len(aroon_data)):
        current_date = aroon_data.index[i]
        price = aroon_data['close'].iloc[i]
        quantity = capital / price
        if aroon_data['LONG'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonY. Long Entry BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif aroon_data['EXIT_LONG'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonY. Exit Long SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif aroon_data['SHORT'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonY. Short Entry SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif aroon_data['EXIT_SHORT'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(aroon_data):
                print(f"AroonY. Exit Short BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")

    return signals


def signals_obv(data, asset, capital=100, n=50):
    data = data.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low"})
    obv_data = strategy_OBV(data, n=n)
    signals = []
    print("OBV Strategy")
    for i in range(1, len(obv_data)):
        current_date = obv_data.index[i]
        price = obv_data['close'].iloc[i]
        quantity = capital / price
        if obv_data['LONG'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(obv_data):
                print(f"OBV. Long Entry BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif obv_data['EXIT_LONG'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(obv_data):
                print(f"OBV. Exit Long SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif obv_data['SHORT'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(obv_data):
                print(f"OBV. Short Entry SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif obv_data['EXIT_SHORT'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(obv_data):
                print(f"OBV. Exit Short BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")

    return signals


def signals_kama(data, asset, capital=100, n=30, f=3, l=30):
    data = data.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"})
    kama_data = strategy_KAMA(data, n=n, f=f, l=l)
    signals = []
    print("KAMA Strategy")
    for i in range(1, len(kama_data)):
        current_date = kama_data.index[i]
        price = kama_data['close'].iloc[i]
        quantity = capital / price
        if kama_data['LONG'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(kama_data):
                print(f"KAMA. Long Entry BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif kama_data['EXIT_LONG'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(kama_data):
                print(f"KAMA. Exit Long SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif kama_data['SHORT'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(kama_data):
                print(f"KAMA. Short Entry SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        elif kama_data['EXIT_SHORT'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(kama_data):
                print(f"KAMA. Exit Short BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")

    return signals


def signals_stoch(data, asset, capital=100):
    data = data.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low"})
    stochastic_data = strategy_Stocastico(data, k=14)
    signals = []
    print("Stochastic Strategy")
    for i in range(1, len(stochastic_data)):
        current_date = stochastic_data.index[i]
        price = stochastic_data['close'].iloc[i]
        quantity = capital / price
        if stochastic_data['EXIT_LONG'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(stochastic_data):
                print(f"Stochastic. Exit Long SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        if stochastic_data['EXIT_SHORT'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(stochastic_data):
                print(f"Stochastic. Exit Short BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        if stochastic_data['LONG'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(stochastic_data):
                print(f"Stochastic. Long Entry BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        if stochastic_data['SHORT'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(stochastic_data):
                print(f"Stochastic. Short Entry SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")

    return signals


def signals_psar(data, asset, capital=100):
    data = data.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low"})
    psar_data = strategy_PSAR(data)
    signals = []
    print("SAR Strategy")
    for i in range(1, len(psar_data)):
        current_date = psar_data.index[i]
        price = psar_data['close'].iloc[i]
        quantity = capital / price
        if psar_data['EXIT_LONG'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(psar_data):
                print(f"SAR. Exit Long SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        if psar_data['EXIT_SHORT'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(psar_data):
                print(f"SAR. Exit Short BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        if psar_data['LONG'].iloc[i]:
            signals.append((asset, 'BUY', quantity, price, current_date))
            if i == len(psar_data):
                print(f"SAR. Long Entry BUY Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")
        if psar_data['SHORT'].iloc[i]:
            signals.append((asset, 'SELL', quantity, price, current_date))
            if i == len(psar_data):
                print(f"SAR. Short Entry SELL Signal: {asset} on {current_date}, Quantity={quantity}, Price={price}")

    return signals


def execute_trades_aaron_x(ib, trades, capital=100, stop_loss_pct=0.98, take_profit_pct=1.18):
    today = date.today()
    for trade in trades:
        asset, action, _, _, trade_date = trade
        if trade_date != today:
            continue
        try:
            contract = Contract()
            contract.symbol = asset
            contract.secType = 'CRYPTO'
            contract.currency = 'USD'
            contract.exchange = 'PAXOS'
            market_price = ib.reqMktData(contract).last
            quantity = capital / market_price if market_price else 0
            if quantity == 0:
                print(f"Not enough capital for any shares of {asset} or market price unavailable.")
                continue
            main_order = MarketOrder(action, quantity)
            trade_order = ib.placeOrder(contract, main_order)
            stop_loss_price = fill_price * stop_loss_pct if action == 'BUY' else fill_price / stop_loss_pct
            take_profit_price = fill_price * take_profit_pct if action == 'BUY' else fill_price / take_profit_pct
            stop_loss_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
            take_profit_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price)
            ib.placeOrder(contract, stop_loss_order)
            ib.placeOrder(contract, take_profit_order)
            print(f"Orders placed for {asset} - Action: {action}, Quantity={quantity}, Price={market_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Date: {trade_date}")
        except Exception as e:
            print(f"Error executing trade for {asset}: {e}")


def execute_trades_aaron_y(ib, trades, capital=100, stop_loss_pct=0.98, take_profit_pct=1.16):
    today = date.today()
    for trade in trades:
        asset, action, _, _, trade_date = trade
        if trade_date != today:
            continue
        try:
            contract = Contract()
            contract.symbol = asset
            contract.secType = 'CRYPTO'
            contract.currency = 'USD'
            contract.exchange = 'PAXOS'
            market_price = ib.reqMktData(contract).last
            quantity = capital / market_price if market_price else 0
            if quantity == 0:
                print(f"Not enough capital for any shares of {asset} or market price unavailable.")
                continue
            main_order = MarketOrder(action, quantity)
            trade_order = ib.placeOrder(contract, main_order)
            stop_loss_price = fill_price * stop_loss_pct if action == 'BUY' else fill_price / stop_loss_pct
            take_profit_price = fill_price * take_profit_pct if action == 'BUY' else fill_price / take_profit_pct
            stop_loss_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
            take_profit_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price)
            ib.placeOrder(contract, stop_loss_order)
            ib.placeOrder(contract, take_profit_order)
            print(f"Orders placed for {asset} - Action: {action}, Quantity={quantity}, Price={market_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Date: {trade_date}")
        except Exception as e:
            print(f"Error executing trade for {asset}: {e}")

            
def execute_trades_obv(ib, trades, capital=100, stop_loss_pct=0.99, take_profit_pct=1.20):
    today = date.today()
    for trade in trades:
        asset, action, _, _, trade_date = trade
        if trade_date != today:
            continue
        try:
            contract = Contract()
            contract.symbol = asset
            contract.secType = 'CRYPTO'
            contract.currency = 'USD'
            contract.exchange = 'PAXOS'
            market_price = ib.reqMktData(contract).last
            quantity = capital / market_price if market_price else 0
            if quantity == 0:
                print(f"Not enough capital for any shares of {asset} or market price unavailable.")
                continue
            main_order = MarketOrder(action, quantity)
            trade_order = ib.placeOrder(contract, main_order)
            stop_loss_price = fill_price * stop_loss_pct if action == 'BUY' else fill_price / stop_loss_pct
            take_profit_price = fill_price * take_profit_pct if action == 'BUY' else fill_price / take_profit_pct
            stop_loss_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
            take_profit_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price)
            ib.placeOrder(contract, stop_loss_order)
            ib.placeOrder(contract, take_profit_order)
            print(f"Orders placed for {asset} - Action: {action}, Quantity={quantity}, Price={market_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Date: {trade_date}")
        except Exception as e:
            print(f"Error executing trade for {asset}: {e}")
            
            
def execute_trades_kama(ib, trades, capital=100, stop_loss_pct=0.98, take_profit_pct=1.17):
    today = date.today()
    for trade in trades:
        asset, action, _, _, trade_date = trade
        if trade_date != today:
            continue
        try:
            contract = Contract()
            contract.symbol = asset
            contract.secType = 'CRYPTO'
            contract.currency = 'USD'
            contract.exchange = 'PAXOS'
            market_price = ib.reqMktData(contract).last
            quantity = capital / market_price if market_price else 0
            if quantity == 0:
                print(f"Not enough capital for any shares of {asset} or market price unavailable.")
                continue
            main_order = MarketOrder(action, quantity)
            trade_order = ib.placeOrder(contract, main_order)
            stop_loss_price = fill_price * stop_loss_pct if action == 'BUY' else fill_price / stop_loss_pct
            take_profit_price = fill_price * take_profit_pct if action == 'BUY' else fill_price / take_profit_pct
            stop_loss_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
            take_profit_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price)
            ib.placeOrder(contract, stop_loss_order)
            ib.placeOrder(contract, take_profit_order)
            print(f"Orders placed for {asset} - Action: {action}, Quantity={quantity}, Price={market_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Date: {trade_date}")
        except Exception as e:
            print(f"Error executing trade for {asset}: {e}")

            
def execute_trades_stoch(ib, trades, capital=100, stop_loss_pct=0.97, take_profit_pct=1.18):
    today = date.today()
    for trade in trades:
        asset, action, _, _, trade_date = trade
        if trade_date != today:
            continue
        try:
            contract = Contract()
            contract.symbol = asset
            contract.secType = 'CRYPTO'
            contract.currency = 'USD'
            contract.exchange = 'PAXOS'
            market_price = ib.reqMktData(contract).last
            quantity = capital / market_price if market_price else 0
            if quantity == 0:
                print(f"Not enough capital for any shares of {asset} or market price unavailable.")
                continue
            main_order = MarketOrder(action, quantity)
            trade_order = ib.placeOrder(contract, main_order)
            stop_loss_price = fill_price * stop_loss_pct if action == 'BUY' else fill_price / stop_loss_pct
            take_profit_price = fill_price * take_profit_pct if action == 'BUY' else fill_price / take_profit_pct
            stop_loss_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
            take_profit_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price)
            ib.placeOrder(contract, stop_loss_order)
            ib.placeOrder(contract, take_profit_order)
            print(f"Orders placed for {asset} - Action: {action}, Quantity={quantity}, Price={market_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Date: {trade_date}")
        except Exception as e:
            print(f"Error executing trade for {asset}: {e}")
            

def execute_trades_psar(ib, trades, capital=100, stop_loss_pct=0.99, take_profit_pct=1.17):
    today = date.today()
    for trade in trades:
        asset, action, _, _, trade_date = trade
        if trade_date != today:
            continue
        try:
            contract = Contract()
            contract.symbol = asset
            contract.secType = 'CRYPTO'
            contract.currency = 'USD'
            contract.exchange = 'PAXOS'
            market_price = ib.reqMktData(contract).last
            quantity = capital / market_price if market_price else 0
            if quantity == 0:
                print(f"Not enough capital for any shares of {asset} or market price unavailable.")
                continue
            main_order = MarketOrder(action, quantity)
            trade_order = ib.placeOrder(contract, main_order)
            stop_loss_price = fill_price * stop_loss_pct if action == 'BUY' else fill_price / stop_loss_pct
            take_profit_price = fill_price * take_profit_pct if action == 'BUY' else fill_price / take_profit_pct
            stop_loss_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
            take_profit_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price)
            ib.placeOrder(contract, stop_loss_order)
            ib.placeOrder(contract, take_profit_order)
            print(f"Orders placed for {asset} - Action: {action}, Quantity={quantity}, Price={market_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Date: {trade_date}")
        except Exception as e:
            print(f"Error executing trade for {asset}: {e}")
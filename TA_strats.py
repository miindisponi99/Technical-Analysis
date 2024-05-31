import ta

import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import minimize

pd.core.common.is_list_like = pd.api.types.is_list_like

def dati_giornalieri(ticker,data_inizio,data_fine):
    """
    Funzione per il caricamento storico prezzi Yahoo
    """
    df = yf.download(ticker, data_inizio,data_fine, auto_adjust = True)
    
    return df


def run_backtest(db_strategia, capitale_ini, sl, tp):
    balance = capitale_ini
    pnl = 0
    position = 0
    stop_loss_lvl = sl
    take_profit_lvl = tp
    last_signal = 'hold'
    last_price = 0
    c = 0

    db_stats=db_strategia.copy()
    db_stats['Ritorno_perc'] = np.nan
    db_stats['BUY_HOLD'] = balance*(db_stats.close / db_stats.open.iloc[0])
    db_stats['RIT_BUY_HOLD'] = (db_stats.close / db_stats.close.shift(1))-1
    db_stats.loc[db_stats.index[0], 'LONG'] = False
    db_stats.loc[db_stats.index[0], 'SHORT'] = False

    for index, row in db_stats.iterrows():
        if row.EXIT_LONG and last_signal == 'long':
            db_stats.loc[index,'Data Fine T']=row.name
            db_stats.loc[index,'Num_Giorni']= c
            db_stats.loc[index,'Exit_Long']=1
            pnl = (row.open - last_price) * position
            db_stats.loc[index,'P&L']=pnl
            db_stats.loc[index,'Ritorno_perc']=(row.open / last_price - 1) * 100
            balance = balance + row.open * position
            position = 0
            last_signal = 'hold'
            c = 0
        elif row.EXIT_SHORT and last_signal == 'short':
            db_stats.loc[index,'Data Fine T']=row.name
            db_stats.loc[index,'Num_Giorni']= c
            db_stats.loc[index,'Exit_Short']=1
            pnl = (row.open - last_price) * position
            db_stats.loc[index,'P&L']=pnl
            db_stats.loc[index,'Ritorno_perc']=(last_price / row.open - 1) * 100
            balance = balance + pnl
            position = 0
            last_signal = 'hold'
            c = 0
        if row.LONG and last_signal != 'long':
            last_signal = 'long'
            last_price = row.open
            db_stats.loc[index,'Entry_Long']=1
            db_stats.loc[index,'Entry_price_L']=last_price
            db_stats.loc[index,'Data inizio T']=row.name
            db_stats.loc[index,'Trade_side']='long'
            position = int(balance / row.open)
            db_stats.loc[index,'Posizione']=position
            cost = position * row.open
            balance = balance - cost
            c = 0
        elif row.SHORT and last_signal != 'short':
            last_signal = 'short'
            last_price = row.open
            db_stats.loc[index,'Entry_Short']=1
            db_stats.loc[index,'Entry_price_S']=last_price
            db_stats.loc[index,'Data inizio T']=row.name
            db_stats.loc[index,'Trade_side']='short'
            position = int(balance / row.open) * -1
            db_stats.loc[index,'Posizione']=position
            c = 0
        if last_signal == 'long' and (row.low / last_price - 1) * 100 <= stop_loss_lvl:
            c = c + 1
            db_stats.loc[index,'Data Fine T']=row.name
            db_stats.loc[index,'Num_Giorni']= c
            stop_loss_price = last_price + round(last_price * (stop_loss_lvl / 100), 4)
            if stop_loss_price > row.low and stop_loss_price < row.high:
                db_stats.loc[index,'SL_price']=stop_loss_price
                db_stats.loc[index,'Exit_SL']=1
                pnl = (stop_loss_price - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(stop_loss_price / last_price - 1) * 100
                balance = balance + stop_loss_price * position
                position = 0
                last_signal = 'hold'
                c = 0
            else:
                db_stats.loc[index,'SL_price']=row.open
                db_stats.loc[index,'Exit_SL']=1
                pnl = (row.open - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(row.open / last_price - 1) * 100
                balance = balance + row.open * position
                position = 0
                last_signal = 'hold'
                c = 0
        elif last_signal == 'short' and (1- row.high/last_price ) * 100 <= stop_loss_lvl:
            c = c + 1
            db_stats.loc[index,'Data Fine T']=row.name
            db_stats.loc[index,'Num_Giorni']= c
            stop_loss_price = last_price - round(last_price * (stop_loss_lvl / 100), 4)
            if stop_loss_price > row.low and stop_loss_price < row.high:
                db_stats.loc[index,'SL_price']=stop_loss_price
                db_stats.loc[index,'Exit_SL']=1
                pnl = (stop_loss_price - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(last_price / stop_loss_price - 1) * 100
                balance = balance + pnl
                position = 0
                last_signal = 'hold'
                c = 0
            else:
                db_stats.loc[index,'SL_price']=row.open
                db_stats.loc[index,'Exit_SL']=1
                pnl = (row.open - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(last_price / row.open - 1) * 100
                balance = balance + pnl
                position = 0
                last_signal = 'hold'
                c = 0
        if last_signal == 'long' and (row.high / last_price - 1) * 100 >= take_profit_lvl:
            c = c + 1
            db_stats.loc[index,'Data Fine T']=row.name
            db_stats.loc[index,'Num_Giorni']= c
            take_profit_price = last_price + round(last_price * (take_profit_lvl / 100), 4)
            if take_profit_price > row.low and take_profit_price < row.high:
                db_stats.loc[index,'TP_price']=take_profit_price
                db_stats.loc[index,'Exit_TP']=1
                pnl = (take_profit_price - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(take_profit_price / last_price - 1) * 100
                balance = balance + take_profit_price * position
                position = 0
                last_signal = 'hold'
                c = 0
            else:
                db_stats.loc[index,'TP_price']=row.open
                db_stats.loc[index,'Exit_TP']=1
                pnl = (row.open - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(row.open / last_price - 1) * 100
                balance = balance + row.open * position
                position = 0
                last_signal = 'hold'
                c = 0
        elif last_signal == 'short' and (1- row.low /last_price ) * 100 >= take_profit_lvl:
            c = c + 1
            db_stats.loc[index,'Data Fine T']=row.name
            db_stats.loc[index,'Num_Giorni']= c
            take_profit_price = last_price - round(last_price * (take_profit_lvl / 100), 4)
            if take_profit_price > row.low and take_profit_price < row.high:
                db_stats.loc[index,'TP_price']=take_profit_price
                db_stats.loc[index,'Exit_TP']=1
                pnl = (take_profit_price - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(last_price / take_profit_price - 1) * 100
                balance = balance + pnl
                position = 0
                last_signal = 'hold'
                c = 0
            else:
                db_stats.loc[index,'TP_price']=row.open
                db_stats.loc[index,'Exit_TP']=1
                pnl = (row.open - last_price) * position
                db_stats.loc[index,'P&L']=pnl
                db_stats.loc[index,'Ritorno_perc']=(last_price / row.open - 1) * 100
                balance = balance + pnl
                position = 0
                last_signal = 'hold'
                c = 0
        if last_signal == 'hold':
            market_value = balance
        elif last_signal == 'long':
            c = c + 1
            market_value = position * row.close + balance
        else: 
            c = c + 1
            market_value = (row.close - last_price) * position + balance
        db_stats.loc[index,'RIT_CUM']=market_value
              
    return db_stats


def rolling_window_backtest(symbol, start_date, end_date, initial_window, roll_period, strategies, backtest_func, initial_capital):
    db = dati_giornalieri(symbol, start_date, end_date)
    db = db.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low"})
    current_start = start_date
    current_end = start_date + initial_window
    all_reports = {strategy_name: [] for strategy_name in strategies}

    while current_end <= end_date:
        window_db = db[(db.index >= current_start) & (db.index < current_end)]
        for strategy_name, strategy_info in strategies.items():
            strategy_func = strategy_info['func']
            strategy_params = strategy_info.get('params', {})
            strategy_db = strategy_func(window_db, **strategy_params)
            backtest_params = strategy_info.get('backtest_params', {})
            stats = backtest_func(strategy_db, initial_capital, **backtest_params)
            report_current = report(stats, stats['P&L'], stats['RIT_CUM'])
            all_reports[strategy_name].append({
                'report': report_current,
                'RIT_CUM': stats.get('RIT_CUM')
            })
        current_start += roll_period
        current_end += roll_period

    return all_reports


def optimize_portfolio(returns_df, volatility_df, correlation_matrices):
    def sharpe_ratio(weights, returns, covariance_matrix):
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        return -portfolio_return / portfolio_volatility

    def maximize_sharpe_ratio(returns, volatilities, correlation_matrix):
        num_assets = len(returns)
        args = (returns, correlation_matrix)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x - 0.01}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1. / num_assets] * num_assets)
        tol = 1e-5
        result = minimize(sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints, tol=tol)

        return result.x if result.success else None

    optimized_weights = {}
    previous_weights = None
    for date in returns_df.index:
        returns = returns_df.loc[date].values
        volatilities = volatility_df.loc[date].values
        correlation_matrix = correlation_matrices.loc[date].values
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        if np.all(returns == 0) and np.all(volatilities == 0) and previous_weights is not None:
            optimized_weights[date] = previous_weights
            continue
            
        weights = maximize_sharpe_ratio(returns, volatilities, covariance_matrix)
        if weights is not None:
            optimized_weights[date] = weights
            previous_weights = weights
        else:
            optimized_weights[date] = previous_weights if previous_weights is not None else np.array([1. / num_assets] * num_assets)

    return optimized_weights


def plot_equity(*equities,colors):
    """
    Funzione per stampare un'equity line
    """
    plt.figure(figsize=(12, 6), dpi=100)
    
    for equity, color in zip(equities, colors):
        plt.plot(equity[1], color=color, label=equity[0])
    
    plt.xlabel("Tempo")
    plt.ylabel("Profitto/Perdite")
    plt.title('Equity Line')
    plt.xticks(rotation='vertical')
    plt.legend(ncol=3)
    plt.grid(True)
    plt.show()


def plot_equity_by_roll(df_rit_cum, roll_number, strategies, colors):
    plt.figure(figsize=(12, 6))
    for (strategy_name, _), color in zip(strategies.items(), colors):
        data_row = df_rit_cum[(df_rit_cum['Strategy'] == strategy_name) & (df_rit_cum['Roll'] == f"Roll {roll_number}")]
        if not data_row.empty:
            plt.plot(data_row['RIT_CUM'].values[0], label=strategy_name, color=color)
    plt.title(f'Equity Line for {roll_number}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_vix_on_volume(db, vix_db):
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.bar(db.index, db['Volume'], label='ETH-USD Volume', color='orange', alpha=0.5, width=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('ETH-USD Volume', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(vix_db.index, vix_db['Close'], label='VIX', color='blue', linewidth=2)
    ax2.set_ylabel('VIX', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.title('VIX Overlay on ETH-USD Volume')
    plt.tight_layout()
    plt.show()


def plot_bh_on_volume(db, db_stats_wr):
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.bar(db.index, db['Volume'], label='ETH-USD Volume', color='orange', alpha=0.5, width=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('ETH-USD Volume', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(db_stats_wr.index, db_stats_wr['BUY_HOLD'], label='B&H Performance', color='blue', linewidth=2)
    ax2.set_ylabel('B&H Performance', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.title('B&H Performance Overlay on ETH-USD Volume')
    plt.tight_layout()
    plt.show()


def plot_equity_vs_bh(equity1,equity2):
    """
    Funzione per stampare due equity sovrapposte
    """
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(equity1, color='green',label='Equity line')
    plt.plot(equity2, color='black',label='Buy & Hold')
    plt.xlabel("Tempo")
    plt.ylabel("Profitto/Perdite")
    plt.title('Equity Line vs Buy & Hold')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    
    return


def drawdown(equity):
    """
    Funzione che calcola il drawdown data un'equity line
    """
    maxvalue = equity.expanding(0).max()
    drawdown = equity - maxvalue
    dd_serie = pd.Series(drawdown, index = equity.index)
    
    return dd_serie


def plot_drawdown(equity,color):
    """
    Funzione per graficare la curva di draw down
    """
    dd = drawdown(equity)
    plt.figure(figsize = (12, 6), dpi = 100)
    plt.plot(dd, color = color)
    plt.fill_between(dd.index, 0, dd, color = color)
    plt.xlabel("Tempo")
    plt.ylabel("Profitto/Perdite")
    plt.title('Draw Down')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    
    return


def profit(equity):
    """
    calcola il profitto della strategia buy & hold
    """
    
    return round(equity[-1]-equity[0],0)


def profit_bh(dataF):
    """
    calcola il profitto del nostro TS
    """
    
    return round(dataF['BUY_HOLD'][-1]-dataF['BUY_HOLD'][0],0)
    

def num_operazioni(operations):
    """
    calcola il numero di operazioni eseguite
    """
    
    return operations.count()


def rendimento_medio(operations):
    """
    calcola il rendimento medio del nostro TS che può essere positivo o negativo
    """
    
    return round(operations.mean(),0)


def max_draw_down(equity):
    """
    Funzione che calcola il massimo draw down
    """
    dd = drawdown(equity)
    
    return round(dd.min(),0)


def max_draw_down_perc(equity):
    """
    Funzione che calcola il massimo draw down percentuale
    """
    maxvalue = equity.expanding(0).max()
    drawdown_pct = (equity/maxvalue-1)*100
    dd_pct = pd.Series(drawdown_pct, index = equity.index)
    
    return round(dd_pct.min(),0)


def avg_dd_nozero(equity):
    """
    calcola la media del draw down storico non considerando valori nulli
    """
    dd = drawdown(equity)
    
    return round(dd[dd < 0].mean(),0)


def perdita_media(operations):
    """
    calcola la perdita media delle operazioni
    """
    
    return round(operations[operations < 0].mean(),0)


def perdita_max(operations):
    """
    calcola la perdita massima
    """
    
    return round(operations.min(),0)


def perdita_max_data(operations):
    """
    calcola la data della perdita massima
    """
    
    return operations.idxmin()


def guadagno_medio(operations):
    
    return round(operations[operations > 0].mean(),0)


def guadagno_max(operations):
    
    return round(operations.max(),0)


def guadagno_max_data(operations):
    """
    calcola la data del guadagno massimo
    """
    
    return operations.idxmax()


def profitto_lordo(operations):
    
    return round(operations[operations > 0].sum(),0)


def perdita_lorda(operations):
    
    return round(operations[operations <= 0].sum(),0)


def profit_factor(operations):
    a = profitto_lordo(operations)
    b = perdita_lorda(operations)
    if b != 0:
        return round(abs(a / b), 2)
    else:
        return round(abs(a / 0.00000001), 2)

    
def percent_win(operations):
    
    return round((operations[operations > 0].count() / operations.count() * 100),2)


def reward_risk_ratio(operations):
    if operations[operations <= 0].mean() != 0:
        return round((operations[operations > 0].mean() / -operations[operations <= 0].mean()),2)
    else:
        return np.inf


def annualize_returns(open_equity):
    """
    Calculate the annualized returns from periodic returns
    """
    returns = open_equity.pct_change().dropna()
    compounded_growth = (1 + returns).prod()
    n_periods = len(returns)
    annualized_return = compounded_growth**(365 / n_periods) - 1
    
    return annualized_return


def annualize_volatility(open_equity):
    """
    Calculate the annualized volatility from periodic returns
    """
    returns = open_equity.pct_change().dropna()

    return returns.std() * np.sqrt(365)


def sharpe_ratio_A(open_equity):
    ann_ret = annualize_returns(open_equity)
    ann_vol = annualize_volatility(open_equity)

    return round(ann_ret/ann_vol if ann_vol != 0 else 0,2)


def sharpe_ratio_BH(returns):
    compounded_growth = (1 + returns).prod()
    n_periods = len(returns)
    ann_ret = compounded_growth**(365 / n_periods) - 1
    ann_vol = returns.std() * np.sqrt(365)

    return round(ann_ret/ann_vol if ann_vol != 0 else 0,2)


def istogrammi_annuali(operations):
    yearly = operations.resample('A').sum()
    colors = pd.Series()
    colors = yearly.apply(lambda x: "green" if x > 0 else "red")
    n_groups = len(yearly)
    plt.subplots(figsize=(12, 6), dpi=100)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1
    rects1 = plt.bar(index,
                     yearly,
                     bar_width,
                     alpha=opacity,
                     color=colors,
                     label='Statistiche annuali')
    plt.xlabel('Anni')
    plt.ylabel('Profitto - Perdite')
    plt.title('Profitto-Perdite annuali')
    plt.xticks(index, yearly.index.year, rotation=90)
    plt.grid(True)
    plt.show()
    
    return


def istogrammi_mensili(operations):
    monthly = pd.DataFrame(operations.fillna(0)).resample('M').sum()
    monthly['Mese'] = monthly.index.month
    biasMonthly = []
    months = []
    for month in range(1, 13):
        months.append(month)
    for month in months:
        biasMonthly.append(monthly[(monthly['Mese'] == month)].mean())
    biasMonthly = pd.DataFrame(biasMonthly)
    column = biasMonthly.columns[0]
    colors = pd.Series()
    colors = biasMonthly[column].apply(lambda x: "green" if x > 0 else "red")
    n_groups = len(biasMonthly)
    plt.subplots(figsize=(12, 6), dpi=100)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1
    rects1 = plt.bar(index,
                     biasMonthly[column],
                     bar_width,
                     alpha=opacity,
                     color=colors,
                     label='Statistiche annuali')
    plt.xlabel('Mesi')
    plt.ylabel('Profitto-Perdita media')
    plt.title('Profitto-Perdita media mensile')
    months_names = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre",
                    "Ottobre", "Novembre", "Dicembre"]
    plt.xticks(index, months_names, rotation=45)
    plt.grid(True)
    plt.show()
    
    return


def plot_equity_heatmap(operations,annotations):
    monthly = operations.resample('M').sum()
    toHeatMap = pd.DataFrame(monthly)
    toHeatMap["Anno"] = toHeatMap.index.year
    toHeatMap["Mese"] = toHeatMap.index.month
    Show = toHeatMap.groupby(by=['Anno','Mese']).sum().unstack()
    Show.columns = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre",
                    "Ottobre", "Novembre", "Dicembre"]
    plt.figure(figsize=(12,6),dpi=100)
    sns.heatmap(Show, cmap="RdYlGn", linecolor="white", linewidth=0.1, annot=annotations, 
                vmin=-max(monthly.min(),monthly.max()), vmax=monthly.max())
    
    return


def var_historic (open_equity, level):
    """
    Compute Value-at-Risk Historic
    """
    returns = open_equity.pct_change().dropna()

    return -np.percentile(returns,level)


def cvar_historic(open_equity, level):
    """
    Compute the Conditional Value-at-Risk of Series/DataFrame
    """
    returns = open_equity.pct_change().dropna()
    is_beyond = returns <= -var_historic(open_equity, level=level)

    return -returns[is_beyond].mean()


def sortino_ratio(open_equity):
    """
    Compute the annualized Sortino Ratio of a set of returns
    """
    returns = open_equity.pct_change().dropna()
    ann_ret = annualize_returns(open_equity)
    neg_ops = returns[returns<0]
    ann_vol = neg_ops.std() * np.sqrt(365)
    sortino = round(ann_ret/ann_vol if ann_vol !=0 else 0,2)

    return sortino


def calmar_ratio(open_equity):
    """
    Compute the annualized Calmar Ratio of a set of returns
    """
    ann_ret = annualize_returns(open_equity)
    max_dd = -max_draw_down_perc(open_equity)/100
    
    return ann_ret/max_dd if max_dd != 0 else 0


def burke_ratio(open_equity, modified = False):
    """
    Compute the annualized Burke Ratio of a set of returns
    If "modified" is True, then the modified Burke Ratio is returned
    """
    ann_ret = annualize_returns(open_equity)
    returns = open_equity.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdowns = (cumulative_returns / peak) - 1
    mean_dd = np.sqrt(np.mean(drawdowns**2))
    drawdowns_squared_sum = np.sqrt(np.sum(drawdowns ** 2))
    if modified == False:
        bk_ratio = ann_ret/drawdowns_squared_sum if drawdowns_squared_sum !=0 else 0
    else:
        bk_ratio = ann_ret/mean_dd if mean_dd !=0 else 0
    
    return bk_ratio


def report(dataF, operations, open_equity, show_charts=False, level=5):
    var_h = round(var_historic(open_equity, level) * 100, 2)
    cvar_h = round(cvar_historic(open_equity, level) * 100, 2)
    sortino = sortino_ratio(open_equity)
    calmar = round(calmar_ratio(open_equity), 2)
    burke = round(burke_ratio(open_equity), 2)
    modified_burke = round(burke_ratio(open_equity, modified=True), 2)
    trade_counts = dataF.groupby('Trade_side').count()['Posizione']
    operazioni_long = trade_counts.get('long', 0)
    operazioni_short = trade_counts.get('short', 0)

    report_dict = {
        "Ritorno Ann": round(annualize_returns(open_equity)*100,2),
        "Volatilità Ann": round(annualize_volatility(open_equity)*100,2),
        "Profitto": profit(open_equity),
        "Profitto Buy&Hold": profit_bh(dataF),
        "Operazioni Chiuse": num_operazioni(operations),
        "Operazioni Long": operazioni_long,
        "Operazioni Short": operazioni_short,
        "Rendimento Medio": rendimento_medio(operations),
        "Rendimento Medio %": round(dataF['Ritorno_perc'].mean(), 2),
        "Dev. Std Rend. Med": round(dataF['Ritorno_perc'].std(), 2),
        "Profit Factor": profit_factor(operations),
        "Profitto Lordo": profitto_lordo(operations),
        "Perdita Lorda": perdita_lorda(operations),
        "% Trade Vincenti": percent_win(operations),
        "% Trade Perdenti": 100 - percent_win(operations),
        "Reward Risk Ratio": reward_risk_ratio(operations),
        "Guadagno Massimo": guadagno_max(operations),
        "Guadagno Massimo Data": guadagno_max_data(operations),
        "Guadagno Medio": guadagno_medio(operations),
        "Guadagno % Medio": round(dataF[dataF['Ritorno_perc'] > 0]['Ritorno_perc'].mean(), 2),
        "Dev. Std Guadagni": round(dataF[dataF['Ritorno_perc'] > 0]['Ritorno_perc'].std(), 2),
        "Perdita Massima": perdita_max(operations),
        "Perdita Massima Data": perdita_max_data(operations),
        "Perdita Media": perdita_media(operations),
        "Perdita % Media": round(dataF[dataF['Ritorno_perc'] < 0]['Ritorno_perc'].mean(), 2),
        "Dev. Std Perdite": round(dataF[dataF['Ritorno_perc'] < 0]['Ritorno_perc'].std(), 2),
        "Avg Draw Down": avg_dd_nozero(open_equity),
        "Max Draw Down": max_draw_down(open_equity),
        "Max Draw Down %": max_draw_down_perc(open_equity),
        "Sharpe Ratio": sharpe_ratio_A(open_equity),
        "Sharpe Ratio B&H": sharpe_ratio_BH(dataF['RIT_BUY_HOLD']),
        "VaR Historic": var_h,
        "CVaR Historic": cvar_h,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Burke Ratio": burke,
        "Modified Burke Ratio": modified_burke
    }
    
    if show_charts:
        plot_equity_vs_bh(open_equity, dataF['BUY_HOLD'])
        plot_drawdown(open_equity, "red")
        istogrammi_annuali(operations)
        istogrammi_mensili(operations)
        plot_equity_heatmap(operations, False)

    return report_dict


def optimize_strategy_and_show_chart(df, strategy_function, param_grid, initial_capital):
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
    results = []
    
    for params in param_combinations:
        temp_df = strategy_function(df, **params)
        result = run_backtest(temp_df, initial_capital, params['sl'], params['tp'])
        results.append({
            **params,
            'Profit': profit(result['RIT_CUM']),
            'Profit Factor': profit_factor(result['P&L']),
            'Sharpe Ratio': sharpe_ratio_A(result['RIT_CUM']),
            'Max Draw Down %': max_draw_down_perc(result['RIT_CUM'])
        }) 
    results_df = pd.DataFrame(results)
    best_params = results_df.sort_values('Sharpe Ratio', ascending=False).iloc[0]
    results_df_sorted = results_df.sort_values(by=['tp', 'sl'])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(results_df_sorted['tp'], results_df_sorted['sl'], results_df_sorted['Sharpe Ratio'], cmap='viridis', linewidth=0.2, antialiased=True)
    ax.set_xlabel('Take Profit')
    ax.set_ylabel('Stop Loss')
    ax.set_zlabel('Sharpe Ratio')
    plt.title('Optimization of Strategy Parameters')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()
    
    return best_params


def strategy_AROON_Xcross(df, **kwargs):
    n = kwargs.get('n', 20)
    data = df.copy()
    aroon = ta.trend.AroonIndicator(high=data['high'], low=data['low'], window=n)
    data['Aroon_Up'] = aroon.aroon_up()
    data['Aroon_Down'] = aroon.aroon_down()
    data['DIFF'] = data['Aroon_Up'] - data['Aroon_Down']
    data['DIFF_PREV'] = data['DIFF'].shift(1)
    data['LONG'] = (data['DIFF'] > 0) & (data['DIFF_PREV'] < 0)
    data['EXIT_LONG'] = (data['DIFF'] < 0) & (data['DIFF_PREV'] > 0)   
    data['SHORT'] = (data['DIFF'] < 0) & (data['DIFF_PREV'] > 0)
    data['EXIT_SHORT'] = (data['DIFF'] > 0) & (data['DIFF_PREV'] < 0)
    data['LONG'] = data['LONG'].shift(1)
    data['EXIT_LONG'] = data['EXIT_LONG'].shift(1)
    data['SHORT'] = data['SHORT'].shift(1)
    data['EXIT_SHORT'] = data['EXIT_SHORT'].shift(1)

    return data


def strategy_AROON_Ycross(df, **kwargs):
    n = kwargs.get('n', 20)
    lvl=kwargs.get('livello', 75)
    data = df.copy()
    aroon = ta.trend.AroonIndicator(high=data['high'], low=data['low'], window=n)
    data['Aroon_Up'] = aroon.aroon_up()
    data['Aroon_Down'] = aroon.aroon_down()
    data['DIFF_U']=data.Aroon_Up-lvl
    data['DIFF_U_PREV']=data.DIFF_U.shift(1)
    data['DIFF_D']=data.Aroon_Down-lvl
    data['DIFF_D_PREV']=data.DIFF_D.shift(1)
    data['LONG'] = (data.DIFF_U>0)&(data.DIFF_U_PREV<0)&(data.Aroon_Down<lvl)
    data['EXIT_LONG'] = (data.DIFF_U<0)&(data.DIFF_U_PREV>=0)
    data['SHORT'] = (data.DIFF_D>0)&(data.DIFF_D_PREV<0)&(data.Aroon_Up<lvl)
    data['EXIT_SHORT'] = (data.DIFF_D<0)&(data.DIFF_D_PREV>=0)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_BollingerBands(df, **kwargs):
    n = kwargs.get('n', 20)
    n_rng = kwargs.get('n_rng', 2)
    data = df.copy()
    boll = ta.volatility.BollingerBands(data.close, n, n_rng)
    
    data['BOLL_LBAND'] = boll.bollinger_lband()
    data['BOLL_UBAND'] = boll.bollinger_hband()
    data['SMA'] = boll.bollinger_mavg()
    data['CLOSE_PREV'] = data.close.shift(1)
    data['LONG'] = (data.close > data.BOLL_LBAND) & (data.CLOSE_PREV <= data.BOLL_LBAND)&(data.close < data.SMA)
    data['EXIT_LONG'] = (data.close > data.SMA) & (data.CLOSE_PREV <= data.SMA)
    data['SHORT'] = (data.close < data.BOLL_UBAND) & (data.CLOSE_PREV >= data.BOLL_UBAND)&(data.close > data.SMA)
    data['EXIT_SHORT'] = (data.close < data.SMA) & (data.CLOSE_PREV >= data.SMA)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_CCI(df, **kwargs):
    n_cci = kwargs.get('n_cci', 20)
    n_sma = kwargs.get('n_sma', 20)
    lvl= kwargs.get('livello', 0)
    data = df.copy()
    cci = ta.trend.CCIIndicator(data.high,data.low,data.close, n_cci)
    sma = ta.trend.SMAIndicator(data.close, n_sma)
    data['CCI'] = cci.cci().round(2)
    data['SMA'] = sma.sma_indicator().round(2)
    data['LVL'] = lvl                            
    data['DIFF'] = data.CCI-lvl
    data['DIFF_PREV'] = data.DIFF.shift(1)
    data['LONG'] = (data.close>data.SMA) & (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['SHORT'] = (data.close<data.SMA) & (data.DIFF < -2*lvl) & (data.DIFF_PREV >= -2*lvl)
    data['EXIT_SHORT'] = (data.DIFF > -2*lvl) & (data.DIFF_PREV <= -2*lvl)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)
    
    return data


def strategy_CCI_2(df, **kwargs):
    n_cci = kwargs.get('n_cci', 20)
    n_sma = kwargs.get('n_sma', 20)
    lvl= kwargs.get('livello', 0)
    data = df.copy()
    cci = ta.trend.CCIIndicator(data.high,data.low,data.close, n_cci)
    sma = ta.trend.SMAIndicator(data.close, n_sma)
    data['CCI'] = cci.cci().round(2)
    data['SMA'] = sma.sma_indicator().round(2)
    data['LVL'] = lvl
    data['DIFF'] = data.close-data.SMA
    data['DIFF_PREV'] = data.DIFF.shift(1)
    data['LONG'] = (data.CCI > lvl) & (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['SHORT'] = (data.CCI < lvl) & (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['EXIT_SHORT'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)
    
    return data


def strategy_HMA_Tether(df, **kwargs):
    n_hma = kwargs.get('n_hma', 20)
    n_tether = kwargs.get('n_tether', 20)
    data = df.copy()
    p=int(n_hma/2)
    radq_n= int(np.sqrt(n_hma))
    pesi=np.linspace(0, n_hma, n_hma)
    pesi_p=np.linspace(0, p, p)
    ps=np.linspace(0, radq_n, radq_n)
    data['w_a'] = 2*data['close'].rolling(p).apply(lambda x: np.sum(pesi_p*x) / np.sum(pesi_p))
    data['w_b'] = data['close'].rolling(n_hma).apply(lambda x: np.sum(pesi*x) / np.sum(pesi))
    data['w_c'] = data['w_a']-data['w_b']
    data['HMA'] = data['w_c'].rolling(radq_n).apply(lambda x: np.sum(ps*x) / np.sum(ps))
    data['massimoass'] = data['high'].rolling(n_tether).max()
    data['minimoass'] = data['low'].rolling(n_tether).min()
    data['TetherLine']= ((data['massimoass']+data['minimoass'])/2)                                          
    data['DIFF'] = data.close-data.TetherLine
    data['DIFF_PREV'] = data.DIFF.shift(1)
    data['LONG'] = (data.DIFF>0) & (data.DIFF_PREV <= 0) & (data.HMA-data.HMA.shift(1) >= 0)& (data.open.shift(-1)-data.TetherLine > 0)
    data['EXIT_LONG'] = (data.DIFF<0) & (data.DIFF_PREV >= 0)                                                              
    data['SHORT'] = (data.DIFF<0) & (data.DIFF_PREV >= 0) & (data.HMA-data.HMA.shift(1) <= 0)& (data.open.shift(-1)-data.TetherLine < 0)
    data['EXIT_SHORT'] = (data.DIFF>0) & (data.DIFF_PREV <= 0)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)
    
    return data


def strategy_Ichimoku(df, **kwargs):
    n_conv = kwargs.get('n_conv', 9)
    n_base = kwargs.get('n_base', 26)
    n_span_b = kwargs.get('n_span_b', 26)
    data = df.copy()
    ichmoku = ta.trend.IchimokuIndicator(data.high, data.low, n_conv, n_base, n_span_b)
    data['BASE'] = ichmoku.ichimoku_base_line().round(2)
    data['CONV'] = ichmoku.ichimoku_conversion_line().round(2)
    #data['SPAN_A'] = ichmoku.ichimoku_a().round(2)
    #data['SPAN_B'] = ichmoku.ichimoku_b().round(2)
    data['SPAN_A'] = ichmoku.ichimoku_a().round(2).shift(n_base)
    data['SPAN_B'] = ichmoku.ichimoku_b().round(2).shift(n_base)
    data['Chikou'] = data.close.shift(-n_base)
    data['DIFF'] = data['CONV'] - data['BASE']
    data['DIFF_PREV'] = data.DIFF.shift(1)
    data['LONG'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0) & (data.close>data['SPAN_A'])& (data.close>data['SPAN_B'])
    data['EXIT_LONG'] = (data.close<data['SPAN_B'])
    data['SHORT'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0) & (data.close<data['SPAN_A'])& (data.close<data['SPAN_B'])
    data['EXIT_SHORT'] = (data.close>data['SPAN_A'])
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_KAMA(df, **kwargs):
    n = kwargs.get('n', 10)
    f = kwargs.get('f', 2)
    l = kwargs.get('l', 30)
    data = df.copy()
    data['kama']= ta.momentum.KAMAIndicator(data.close,n,f,l).kama()
    data['CLOSE_PREV'] = data.close.shift(1)
    data['LONG'] = (data.close > data.kama) & (data.CLOSE_PREV <= data.kama)
    data['EXIT_LONG'] = (data.close < data.kama) & (data.CLOSE_PREV >= data.kama)
    data['SHORT'] = (data.close < data.kama) & (data.CLOSE_PREV >= data.kama)
    data['EXIT_SHORT'] = (data.close > data.kama) & (data.CLOSE_PREV <= data.kama)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)
    
    return data


def strategy_Keltner(df, **kwargs):
    n = kwargs.get('n', 20)
    k = kwargs.get('k')
    data = df.copy()
    k_band = ta.volatility.KeltnerChannel(data.high, data.low, data.close, n)
    
    data['K_BAND_UB'] = (1+k/100)*k_band.keltner_channel_hband().round(4)
    data['K_BAND_LB'] = (1-k/100)*k_band.keltner_channel_lband().round(4)
    data['M_BAND'] = k_band.keltner_channel_mband().round(4)
    data['CLOSE_PREV'] = data.close.shift(1)
    data['LONG'] = (data.close > data.K_BAND_LB) & (data.CLOSE_PREV <= data.K_BAND_LB)&(data.close < data.M_BAND)
    data['EXIT_LONG'] = (data.close >= data.M_BAND) & (data.CLOSE_PREV < data.M_BAND)
    data['SHORT'] = (data.close < data.K_BAND_UB) & (data.CLOSE_PREV >= data.K_BAND_UB)&(data.close > data.M_BAND)
    data['EXIT_SHORT'] = (data.close <= data.M_BAND) & (data.CLOSE_PREV > data.M_BAND)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_Keltner_break(df, **kwargs):
    n = kwargs.get('n', 20)
    k = kwargs.get('k')
    data = df.copy()
    k_band = ta.volatility.KeltnerChannel(data.high, data.low, data.close, n)

    data['K_BAND_UB'] = (1+k/100)*k_band.keltner_channel_hband().round(4)
    data['K_BAND_LB'] = (1-k/100)*k_band.keltner_channel_lband().round(4)
    data['M_BAND'] = k_band.keltner_channel_mband().round(4)
    data['CLOSE_PREV'] = data.close.shift(1)
    data['LONG'] = (data.close > data.K_BAND_UB) & (data.CLOSE_PREV <= data.K_BAND_UB)
    data['EXIT_LONG'] = (data.close <= data.K_BAND_UB) & (data.CLOSE_PREV > data.K_BAND_UB)
    data['SHORT'] = (data.close < data.K_BAND_LB) & (data.CLOSE_PREV >= data.K_BAND_LB)
    data['EXIT_SHORT'] = (data.close >= data.K_BAND_LB) & (data.CLOSE_PREV < data.K_BAND_LB)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_MACD(df, **kwargs):
    n_slow = kwargs.get('n_slow', 26)
    n_fast = kwargs.get('n_fast', 12)
    n_sign = kwargs.get('n_sign', 9)
    data = df.copy()
    macd = ta.trend.MACD(data.close, n_slow, n_fast, n_sign)
    
    data['MACD_LINE'] = macd.macd().round(4)
    data['SIG_LINE'] = macd.macd_signal().round(4)                                     
    data['MACD_DIFF'] = macd.macd_diff().round(4)
    data['MACD_DIFF_PREV'] = data.MACD_DIFF.shift(1)
    data['LONG'] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)
    data['EXIT_LONG'] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)
    data['SHORT'] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)
    data['EXIT_SHORT'] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_OBV(df, **kwargs):
    n = kwargs.get('n', 30)
    data = df.copy()
    data['OBV']= ta.volume.OnBalanceVolumeIndicator(data.close,data.Volume).on_balance_volume()
    ema = ta.trend.EMAIndicator(data.OBV, n)
    data['OBV_ema'] = ema.ema_indicator().round(2)
    data['DIFF'] = data['OBV'] - data['OBV_ema']
    data['DIFF_PREV'] = data.DIFF.shift(1)
    data['LONG'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['SHORT'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['EXIT_SHORT'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_Stocastico(df, **kwargs):
    k = kwargs.get('k', 20)
    d = kwargs.get('d', 5)
    dd = kwargs.get('dd', 3)
    data = df.copy()
    sto = ta.momentum.StochasticOscillator(data.high, data.low, data.close, k, d)

    data['K'] = sto.stoch().round(4)
    data['D'] = sto.stoch_signal().round(4)
    ma = ta.trend.SMAIndicator(data.D, dd)
    data['DD'] = ma.sma_indicator().round(4)
    data['DIFF'] = data['D'] - data['DD']
    data['DIFF_PREV'] = data.DIFF.shift(1)
    data['LONG'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['SHORT'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
    data['EXIT_SHORT'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_PSAR(df, **kwargs):
    data = df.copy()
    sar = ta.trend.PSARIndicator(data.high,data.low,data.close)
    
    data['PSAR'] = sar.psar()
    data['Down_Ind'] = sar.psar_down_indicator()
    data['Up_Ind'] = sar.psar_up_indicator()
    data['LONG'] = sar.psar_up_indicator()
    data['EXIT_LONG'] = sar.psar_down_indicator()   
    data['SHORT'] = sar.psar_down_indicator()
    data['EXIT_SHORT'] = sar.psar_up_indicator()
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data


def strategy_WR(df, **kwargs):
    n = kwargs.get('n', 14)
    data = df.copy()
    wr = ta.momentum.WilliamsRIndicator(data.high, data.low, data.close, n)

    data['WR'] = wr.williams_r().round(4)
    data['WR_PREV'] = data.WR.shift(1)
    data['LONG'] = (data.WR > -80) & (data.WR_PREV <= -80)
    data['EXIT_LONG'] = (data.WR < -20) & (data.WR_PREV >= -20)
    data['SHORT'] = (data.WR < -20) & (data.WR_PREV >= -20)
    data['EXIT_SHORT'] = (data.WR > -80) & (data.WR_PREV <= -80)
    data.LONG = data.LONG.shift(1)
    data.EXIT_LONG = data.EXIT_LONG.shift(1)
    data.SHORT = data.SHORT.shift(1)
    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

    return data
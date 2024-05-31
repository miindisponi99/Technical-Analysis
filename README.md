# Technical-Analysis

## Purpose

The purpose of this repository is to provide a comprehensive collection of trading strategies based on technical analysis of Ethereum cryptocurrency. Each strategy is designed to analyze historical financial data and generate buy/sell signals based on various technical indicators. This repository is useful for traders, analysts, and researchers who want to explore, backtest, and compare different trading strategies.

## Strategies Implemented

In this repository, you will find the following trading strategies implemented with detailed explanations:

### Aroon Indicator Strategies:
- **AROON X-Cross**: Generates signals based on the crossover of Aroon Up and Aroon Down indicators.
- **AROON Y-Cross**: Uses a threshold level to generate signals when Aroon indicators cross specified levels.

### Bollinger Bands Strategy:
- Generates signals based on price interactions with Bollinger Bands, including the middle band (SMA) and upper/lower bands.

### Commodity Channel Index (CCI) Strategies:
- **CCI with Level Cross**: Uses the CCI indicator to generate signals when crossing a specified level.
- **CCI and SMA**: Combines CCI and Simple Moving Average (SMA) for signal generation.

### Hull Moving Average (HMA) and Tether Line Strategy:
- Combines HMA with a tether line based on the highest and lowest prices over a specified period.

### Ichimoku Cloud Strategy:
- Uses Ichimoku Cloud components such as conversion line, base line, and span lines to generate signals.

### Kaufman's Adaptive Moving Average (KAMA) Strategy:
- Generates signals based on the relationship between the closing price and KAMA.

### Keltner Channel Strategies:
- **Keltner Channel**: Generates signals based on price interactions with Keltner Channel bands.
- **Keltner Channel Breakout**: Focuses on breakouts above or below the Keltner Channel bands.

### MACD Strategy:
- Uses the Moving Average Convergence Divergence (MACD) indicator to generate signals based on the MACD line and signal line crossovers.

### On-Balance Volume (OBV) Strategy:
- Combines OBV with its Exponential Moving Average (EMA) to generate signals based on volume trends.

### Stochastic Oscillator Strategy:
- Uses the Stochastic Oscillator and its moving average to generate signals based on momentum.

### Parabolic SAR Strategy:
- Generates signals based on the Parabolic SAR indicator, which tracks price reversals.

### Williams %R Strategy:
- Uses the Williams %R indicator to generate signals based on overbought and oversold conditions.

## Files

### All_Strategies Jupyter Notebook
Analyzes all the aforementioned strategies using training, validation, and testing datasets. This notebook serves as the initial step in the strategy development process, providing a comprehensive analysis of each strategy's performance.

### Validation_Params Jupyter Notebook
Further analyzes the best performing strategies identified in the All_strategies notebook. This notebook focuses on fine-tuning and optimizing the parameters of these strategies to enhance their performance.

### Final_Portfolio Jupyter Notebook
Creates a unique portfolio comprising the best performing strategies with optimized parameters. This notebook uses out-of-sample data to evaluate performance and risk metrics, comparing the portfolio to a buy-and-hold portfolio. Backtesting is conducted using a rolling window approach to ensure robustness and reliability of the results.

### IB_strats and IB_Crypto Jupyter Notebook
These two files automate the identification of trade signals and execution of trades using various trading strategies via the Interactive Brokers (IB) API.
The key components include functions for signal generation based on strategies such as Aroon X-Cross, Aroon Y-Cross, On-Balance Volume (OBV), Kaufman's Adaptive Moving Average (KAMA), Stochastic Oscillator, and Parabolic SAR. Additionally, files contain trade execution functions for each strategy, ensuring trades are placed with proper risk management through stop-loss and take-profit orders and the strategy runs daily at market open.


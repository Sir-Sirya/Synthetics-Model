# Synthetics Model - Algorithmic Trading Bot

A live trading system that trades Deriv Synthetic Volatility Indices using machine learning-powered market regime detection and autoregressive return prediction.

## Overview

This bot implements a sophisticated trading strategy combining:
- **Multi-timeframe technical analysis** (Fibonacci levels, ATR, price structure)
- **Gaussian Mixture Model clustering** for market regime identification
- **Regime-conditioned Ridge regression** for return prediction
- **Risk-managed execution** with strict position sizing and equity limits

## Features

### 📊 Multi-Timeframe Data Pipeline
- Real-time data synchronization from MetaTrader5
- Hierarchical price levels: Weekly, Daily (1d/2d/3d/10d/20d), and 15-minute candles
- Fallback mechanisms to handle MT5 cache delays

### 🧠 Feature Engineering
- **Volatility Metrics**: Average True Range (ATR) normalization
- **Price Structure**: Premium/Discount positioning within multi-timeframe ranges
- **Market Microstructure**: Higher-high/Higher-low pattern detection
- **Fibonacci Grid**: OTE midpoint (70.5%) and extension targets (T1, T2, Symmetrical)
- **Trend Indicators**: Macro trend divergence across 10d/20d equilibrium levels

### 🎯 Market Regime Detection
Four distinct market regimes identified via GMM clustering:
- **Consolidation**: Low volatility, tight ranges
- **Retracement**: Counter-trend movement within structure
- **Reversal**: Shift in structural direction
- **Expansion**: High volatility, extended moves

### 🤖 Regime-Conditioned AR Models
- Separate Ridge regression model trained for each market regime
- Inputs: 6 structural features + 3 return lags
- Output: Predicted next-candle return

### ⚖️ Risk Management
- Maximum 2% equity risk per trade
- Dynamic lot sizing based on stop-loss distance
- Position limit: 1 concurrent trade
- Strict broker volume constraints enforcement

## Trading Logic

### Entry Conditions

**BUY Signal:**
- Predicted return > ATR × 0.1
- Market regime in ["Expansion", "Retracement"]
- 3-day premium/discount zone < 0.5 (discount territory)

**SELL Signal:**
- Predicted return < -ATR × 0.1
- Market regime in ["Expansion", "Reversal"]
- 3-day premium/discount zone > 0.5 (premium territory)

### Exit Strategy
- **Stop Loss**: ATR × 2.5
- **Take Profit**: ATR × 4.0

## Installation

### Prerequisites
- Python 3.8+
- MetaTrader5 terminal with Deriv account
- API access enabled

### Setup

1. Clone and navigate to the project:
```bash
cd synthetics_model
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure credentials in `config/settings.json`:
```json
{
  "mt5": {
    "account": "YOUR_ACCOUNT_NUMBER",
    "password": "YOUR_PASSWORD",
    "server": "Deriv-Server"
  }
}
```

## Project Structure

```
synthetics_model/
├── main.py                 # Live trading loop & initialization
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config/
│   └── settings.json      # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── constants.py       # Synthetic symbol definitions
│   ├── data_loader.py     # MT5 data pipeline
│   ├── features.py        # Feature engineering
│   ├── models.py          # GMM clustering & AR models
│   └── execution.py       # Order management
└── notebooks/             # Analysis & backtesting (future)
```

## Module Reference

### `data_loader.py`
- **MT5DataPipeline**: Manages data synchronization and multi-timeframe lookbacks
- **select_target_asset()**: Interactive asset selection UI

### `features.py`
- **FeatureEngineer**: Computes ATR, premium/discount, Fibonacci grids, market structure

### `models.py`
- **MarketRegimeClustering**: Fits GMM and maps clusters to market regime labels
- **RegimeConditionedARModel**: Trains and predicts using regime-specific Ridge models

### `execution.py`
- **MT5ExecutionManager**: Handles position tracking, risk sizing, and order placement

### `constants.py`
- **SyntheticSymbols**: Enum of Deriv volatility indices (V10-V250)

## Usage

Run the live trading bot:
```bash
python main.py
```

The system will:
1. Initialize MT5 connection
2. Train on 30,000 bars of historical data
3. Enter continuous monitoring loop checking for trade signals every 15-minute candle
4. Log all decisions and executions to console

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `magic_number` | 202606 | Unique identifier for bot positions |
| `max_positions` | 1 | Maximum concurrent open positions |
| `risk_percentage` | 2.0 | Equity risk per trade (%) |
| `sl_multiplier` | 2.5 | Stop-loss distance (ATR multiplier) |
| `tp_multiplier` | 4.0 | Take-profit distance (ATR multiplier) |
| `atr_period` | 14 | ATR lookback period |
| `n_regimes` | 4 | Number of market regimes to cluster |

## Known Limitations

- **Position Limit**: Only 1 active trade at a time (conservative risk management)
- **Asset Scope**: Limited to Deriv synthetic indices (no forex/crypto)
- **Data Dependency**: Requires continuous MT5 terminal connection
- **Historical Data**: Needs at least 30,000 bars for initial training
- **No Walk-Forward Validation**: Uses full dataset for training (no out-of-sample testing)

## Future Enhancements

- [ ] Backtesting framework with walk-forward validation
- [ ] Parameter optimization (hyperparameter tuning)
- [ ] Multi-position strategy variant
- [ ] Risk metrics dashboard
- [ ] Position scaling/pyramiding logic
- [ ] Regime probability weighting for entries
- [ ] Real-time performance monitoring

## Troubleshooting

**MT5 Connection Failed**
- Verify MetaTrader5 terminal is running
- Check account credentials in `settings.json`
- Ensure API is enabled in terminal settings

**No Data Returned**
- Confirm symbol is available in Market Watch
- Check MT5 terminal connectivity
- Verify sufficient historical data exists

**Insufficient Regime Samples**
- Model may skip training for regimes with <20 samples
- Increase `historical_bars` or collect more live data

## Contributing

To extend or modify the system:
1. Test changes in `notebooks/` with backtesting scripts
2. Validate against live data before deployment
3. Update `README.md` and configuration schema as needed

## License

Proprietary trading system. Use at your own risk.

## Support

For issues or questions, review logs in the terminal output with `logging.INFO` level enabled.

---

**Last Updated**: 2026-06-13  
**Version**: 1.0.0

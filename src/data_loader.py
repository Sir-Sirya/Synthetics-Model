import logging
import time
from enum import Enum
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticSymbols(Enum):
    """
    Enum representing Deriv Synthetic Indices.
    Values must match the MT5 Market Watch symbols exactly.
    """
    V10 = "Volatility 10 Index"
    V25 = "Volatility 25 Index"
    V50 = "Volatility 50 Index"
    V75 = "Volatility 75 Index"
    V100 = "Volatility 100 Index"
    
    V10_1S = "Volatility 10 (1s) Index"
    V15_1S = "Volatility 15 (1s) Index"
    V25_1S = "Volatility 25 (1s) Index"
    V30_1S = "Volatility 30 (1s) Index"
    V50_1S = "Volatility 50 (1s) Index"
    V75_1S = "Volatility 75 (1s) Index"
    V90_1S = "Volatility 90 (1s) Index"
    V100_1S = "Volatility 100 (1s) Index"
    V150_1S = "Volatility 150 (1s) Index"
    V250_1S = "Volatility 250 (1s) Index"

class MT5DataPipeline:
    def __init__(self, symbol: SyntheticSymbols, base_tf: int):
        self.symbol_str = symbol.value
        self.base_tf = base_tf
        
        if not mt5.initialize():
            raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")
            
        # Programmatically force select symbol in Market Watch window
        mt5.symbol_select(self.symbol_str, True)
        logging.info(f"MT5 Initialized successfully for {self.symbol_str}")

    def get_clean_m15_data(self, n_bars: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol_str, self.base_tf, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No M15 execution data returned from MT5 for {self.symbol_str}.")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_historical_high_low_fallback(self, target_time: datetime, timeframe: int, n_bars: int) -> dict:
        """
        Fallback engine that queries point-in-time rates explicitly per bar 
        if the standard vectorized matrix yields empty slots due to local cache lag.
        """
        # Request an extra bar to confirm handling of the uncompleted active block
        rates = mt5.copy_rates_from(self.symbol_str, timeframe, target_time, n_bars + 1)
        
        if rates is None or len(rates) < n_bars:
            return {"high": np.nan, "low": np.nan}
            
        last_bar_time = pd.to_datetime(rates[-1]['time'], unit='s')
        
        # Discard active or future bars
        if last_bar_time >= target_time:
            valid_rates = rates[:n_bars]
        else:
            valid_rates = rates[-n_bars:]
            
        return {
            "high": float(max(r['high'] for r in valid_rates)),
            "low": float(min(r['low'] for r in valid_rates))
        }

    def build_synchronized_dataset(self, n_bars: int) -> pd.DataFrame:
        """
        Iterates and builds historical arrays mapping static and dynamic multi-lookback 
        hierarchies point-in-time with verified fallback mechanisms to eliminate NaN issues.
        """
        df_base = self.get_clean_m15_data(n_bars)
        
        # Structure arrays
        w1_h, w1_l = [], []
        d1_h1, d1_l1 = [], []
        d2_h2, d2_l2 = [], []
        d3_h3, d3_l3 = [], []
        d10_h10, d10_l10 = [], []
        d20_h20, d20_l20 = [], []
        
        logging.info(f"Processing structural sync for {n_bars} bars...")
        
        for idx, row in df_base.iterrows():
            current_timestamp = row['time']
            
            # 1. Static Weekly Level (Completed Previous Week)
            w1_data = self.get_historical_high_low_fallback(current_timestamp, mt5.TIMEFRAME_W1, 1)
            w1_h.append(w1_data['high'])
            w1_l.append(w1_data['low'])
            
            # 2. Dynamic 1-Day Ago Level
            d1_data = self.get_historical_high_low_fallback(current_timestamp, mt5.TIMEFRAME_D1, 1)
            d1_h1.append(d1_data['high'])
            d1_l1.append(d1_data['low'])
            
            # 3. Dynamic 2-Days Ago Level (Offsets and captures the isolated 1-day bar prior)
            d2_raw = mt5.copy_rates_from(self.symbol_str, mt5.TIMEFRAME_D1, current_timestamp, 3)
            if d2_raw is not None and len(d2_raw) >= 3:
                # Discard index matching current day or future day relative to stamp
                d2_clean = [r for r in d2_raw if pd.to_datetime(r['time'], unit='s') < current_timestamp.replace(hour=0, minute=0, second=0)]
                if len(d2_clean) >= 2:
                    d2_h2.append(float(d2_clean[-2]['high']))
                    d2_l2.append(float(d2_clean[-2]['low']))
                else:
                    d2_h2.append(np.nan); d2_l2.append(np.nan)
            else:
                d2_h2.append(np.nan); d2_l2.append(np.nan)
                
            # 4. Dynamic Multi-Lookback Channels (3d, 10d, 20d)
            d3_data = self.get_historical_high_low_fallback(current_timestamp, mt5.TIMEFRAME_D1, 3)
            d3_h3.append(d3_data['high'])
            d3_l3.append(d3_data['low'])
            
            d10_data = self.get_historical_high_low_fallback(current_timestamp, mt5.TIMEFRAME_D1, 10)
            d10_h10.append(d10_data['high'])
            d10_l10.append(d10_data['low'])
            
            d20_data = self.get_historical_high_low_fallback(current_timestamp, mt5.TIMEFRAME_D1, 20)
            d20_h20.append(d20_data['high'])
            d20_l20.append(d20_data['low'])
            
        # Map back to final dataframe columns
        df_base['static_weekly_high'] = w1_h
        df_base['static_weekly_low'] = w1_l
        df_base['dynamic_prev1d_high'] = d1_h1
        df_base['dynamic_prev1d_low'] = d1_l1
        df_base['dynamic_prev2d_high'] = d2_h2
        df_base['dynamic_prev2d_low'] = d2_l2
        df_base['dynamic_roll_high_3'] = d3_h3
        df_base['dynamic_roll_low_3'] = d3_l3
        df_base['dynamic_roll_high_10'] = d10_h10
        df_base['dynamic_roll_low_10'] = d10_l10
        df_base['dynamic_roll_high_20'] = d20_h20
        df_base['dynamic_roll_low_20'] = d20_l20
        
        return df_base

def select_target_asset() -> SyntheticSymbols:
    print("\n================================")
    print("  SYNTHETIC INDEX ASSET SELECTION")
    print("================================")
    for idx, symbol in enumerate(SyntheticSymbols):
        print(f"[{idx}] {symbol.name} -> {symbol.value}")
    while True:
        try:
            choice = int(input("\nEnter the index number of the symbol you want to analyze: "))
            if 0 <= choice < len(SyntheticSymbols):
                return list(SyntheticSymbols)[choice]
            print("Selection out of range.")
        except ValueError:
            print("Invalid entry. Use numerical values.")

if __name__ == "__main__":
    selected_asset = select_target_asset()
    try:
        pipeline = MT5DataPipeline(selected_asset, mt5.TIMEFRAME_M15)
        # Verify alignment on a 100 bar execution space
        dataset = pipeline.build_synchronized_dataset(100)
        
        print("\n=== SYSTEM LEVELS SYNC MATRIX PREVIEW ===")
        preview_cols = [
            'time', 'close',
            'static_weekly_high', 'static_weekly_low',
            'dynamic_prev1d_high', 'dynamic_prev1d_low',
            'dynamic_prev2d_high', 'dynamic_prev2d_low',
            'dynamic_roll_high_20', 'dynamic_roll_low_20'
        ]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(dataset[preview_cols].tail(5).to_string())
        print("=========================================\n")
    except Exception as e:
        logging.error(f"Process runtime error: {str(e)}")
    finally:
        mt5.shutdown()
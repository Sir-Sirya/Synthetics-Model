import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Computes trailing volatility bounds for normalization."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        matrix = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = matrix.max(axis=1)
        return true_range.rolling(window=self.atr_period).mean()

    def _compute_range_position(self, close: pd.Series, high_level: pd.Series, low_level: pd.Series) -> pd.Series:
        """Maps price location within a dealing range (0.5 = Equilibrium)."""
        range_size = high_level - low_level
        return (close - low_level) / (range_size + 1e-8)

    def _compute_fib_distances(self, df: pd.DataFrame, prefix: str, high_col: pd.Series, low_col: pd.Series):
        """
        Dynamically projects the Fibonacci grid onto the active dealing range
        and measures the close price distance to OTE and expansion profit targets.
        """
        range_size = high_col - low_col
        
        # 1. Optimal Trade Entry (OTE) Midpoint level (70.5%)
        # Projected from both boundaries to maintain structural invariance
        ote_bullish = low_col + (0.705 * range_size)
        ote_bearish = high_col - (0.705 * range_size)
        
        df[f'feat_{prefix}_dist_ote_bull'] = (ote_bullish - df['close']) / (df['atr'] + 1e-8)
        df[f'feat_{prefix}_dist_ote_bear'] = (df['close'] - ote_bearish) / (df['atr'] + 1e-8)
        
        # 2. Institutional Expansion / Profit Taking Target Extensions
        # Target 1 (-27%), Target 2 (-62%), Symmetrical Swing Swings (-100%)
        t1_ext_bull = high_col + (0.27 * range_size)
        t2_ext_bull = high_col + (0.62 * range_size)
        sym_ext_bull = high_col + (1.00 * range_size)
        
        t1_ext_bear = low_col - (0.27 * range_size)
        t2_ext_bear = low_col - (0.62 * range_size)
        sym_ext_bear = low_col - (1.00 * range_size)
        
        # Measure structural proximity to extensions
        df[f'feat_{prefix}_dist_t1_bull'] = (t1_ext_bull - df['close']) / (df['atr'] + 1e-8)
        df[f'feat_{prefix}_dist_t2_bull'] = (t2_ext_bull - df['close']) / (df['atr'] + 1e-8)
        df[f'feat_{prefix}_dist_sym_bull'] = (sym_ext_bull - df['close']) / (df['atr'] + 1e-8)
        
        df[f'feat_{prefix}_dist_t1_bear'] = (df['close'] - t1_ext_bear) / (df['atr'] + 1e-8)
        df[f'feat_{prefix}_dist_t2_bear'] = (df['close'] - t2_ext_bear) / (df['atr'] + 1e-8)
        df[f'feat_{prefix}_dist_sym_bear'] = (df['close'] - sym_ext_bear) / (df['atr'] + 1e-8)

    def generate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point to assemble normalized premium/discount metrics, 
        market structure trends, and Fibonacci symmetry confluences.
        """
        df = df.copy()
        df['atr'] = self.calculate_atr(df)
        
        # Displacement Vectors
        df['feat_displacement'] = (df['close'] - df['open']) / (df['atr'] + 1e-8)
        
        # Premium / Discount Continuums
        df['feat_pd_weekly'] = self._compute_range_position(df['close'], df['static_weekly_high'], df['static_weekly_low'])
        df['feat_pd_roll_3d'] = self._compute_range_position(df['close'], df['dynamic_roll_high_3'], df['dynamic_roll_low_3'])
        df['feat_pd_roll_10d'] = self._compute_range_position(df['close'], df['dynamic_roll_high_10'], df['dynamic_roll_low_10'])
        
        # Market Structure Directionality Matrices
        hh_micro = df['dynamic_prev1d_high'] > df['dynamic_prev2d_high']
        hl_micro = df['dynamic_prev1d_low'] > df['dynamic_prev2d_low']
        lh_micro = df['dynamic_prev1d_high'] < df['dynamic_prev2d_high']
        ll_micro = df['dynamic_prev1d_low'] < df['dynamic_prev2d_low']
        
        df['feat_ms_micro_bullish'] = np.where(hh_micro & hl_micro, 1.0, 0.0)
        df['feat_ms_micro_bearish'] = np.where(lh_micro & ll_micro, 1.0, 0.0)
        
        eq_10d = (df['dynamic_roll_high_10'] + df['dynamic_roll_low_10']) / 2
        eq_20d = (df['dynamic_roll_high_20'] + df['dynamic_roll_low_20']) / 2
        df['feat_ms_macro_trend'] = (eq_10d - eq_20d) / (df['atr'] + 1e-8)
        
        # Fibonacci Symmetry Computations
        self._compute_fib_distances(df, '3d', df['dynamic_roll_high_3'], df['dynamic_roll_low_3'])
        self._compute_fib_distances(df, '10d', df['dynamic_roll_high_10'], df['dynamic_roll_low_10'])
        
        # Machine learning modeling label
        df['target_return'] = df['close'].shift(-1) - df['close']
        
        df.dropna(inplace=True)
        return df

if __name__ == "__main__":
    print("Feature Engineering framework compilation complete. Fibonacci extension layer online.")
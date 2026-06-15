import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketRegimeClustering:
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=random_state, n_init=5)
        self.feature_cols = [
            'feat_displacement', 'feat_pd_weekly', 'feat_pd_roll_3d', 'feat_pd_roll_10d',
            'feat_ms_micro_bullish', 'feat_ms_micro_bearish', 'feat_ms_macro_trend',
            'feat_3d_dist_ote_bull', 'feat_3d_dist_ote_bear',
            'feat_10d_dist_t1_bull', 'feat_10d_dist_t1_bear'
        ]
        self.regime_map = {}

    def _determine_regime_labels(self, df: pd.DataFrame, scaled_features: np.ndarray, cluster_assignments: np.ndarray):
        temp_df = pd.DataFrame(scaled_features, columns=self.feature_cols)
        temp_df['cluster'] = cluster_assignments
        temp_df['raw_displacement'] = df['feat_displacement'].values

        cluster_profiles = temp_df.groupby('cluster').agg({
            'raw_displacement': lambda x: x.abs().mean()
        }).reset_index()

        sorted_profiles = cluster_profiles.sort_values(by='raw_displacement').reset_index(drop=True)

        self.regime_map[sorted_profiles.loc[0, 'cluster']] = "Consolidation"
        self.regime_map[sorted_profiles.loc[1, 'cluster']] = "Retracement"
        self.regime_map[sorted_profiles.loc[2, 'cluster']] = "Reversal"
        self.regime_map[sorted_profiles.loc[3, 'cluster']] = "Expansion"

        for cluster_id, label in self.regime_map.items():
            logging.info(f"GMM Cluster {cluster_id} mapped to Market Cycle: {label}")

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        X = df_copy[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        raw_clusters = self.gmm.fit_predict(X_scaled)
        self._determine_regime_labels(df_copy, X_scaled, raw_clusters)
        
        df_copy['regime_cluster_id'] = raw_clusters
        df_copy['market_regime'] = df_copy['regime_cluster_id'].map(self.regime_map)
        
        probabilities = self.gmm.predict_proba(X_scaled)
        for i in range(self.n_regimes):
            label = self.regime_map.get(i, f"cluster_{i}")
            df_copy[f'prob_regime_{label.lower()}'] = probabilities[:, i]
            
        return df_copy


class RegimeConditionedARModel:
    def __init__(self, alpha: float = 1.0):
        """
        Manages sub-lags generation and trains separate Ridge models 
        conditioned exclusively on the active structural GMM regime.
        """
        self.regimes = ["Consolidation", "Retracement", "Reversal", "Expansion"]
        self.models = {regime: Ridge(alpha=alpha) for regime in self.regimes}
        
        # Core structural feature inputs combined with autoregressive price return lags
        self.base_features = [
            'feat_displacement', 'feat_pd_roll_3d', 'feat_pd_roll_10d', 
            'feat_ms_macro_trend', 'feat_3d_dist_ote_bull', 'feat_3d_dist_ote_bear'
        ]
        self.lag_features = ['lag_ret_1', 'lag_ret_2', 'lag_ret_3']
        self.all_features = self.base_features + self.lag_features

    def prepare_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates historical return lags to fulfill the autoregressive definition."""
        df = df.copy()
        # Compute immediate past execution candle returns
        immediate_return = df['close'] - df['close'].shift(1)
        
        df['lag_ret_1'] = immediate_return.shift(0)  # Most recently completed bar return
        df['lag_ret_2'] = immediate_return.shift(1)
        df['lag_ret_3'] = immediate_return.shift(2)
        
        df.dropna(subset=self.lag_features, inplace=True)
        return df

    def fit(self, df: pd.DataFrame):
        """Fits an individual Ridge pipeline for each structural market environment."""
        df_lagged = self.prepare_lags(df)
        
        for regime in self.regimes:
            regime_data = df_lagged[df_lagged['market_regime'] == regime]
            
            if len(regime_data) < 20:
                logging.warning(f"Insufficient sample size ({len(regime_data)}) to fit AR model for {regime}. Skipping.")
                continue
                
            X = regime_data[self.all_features].values
            y = regime_data['target_return'].values
            
            self.models[regime].fit(X, y)
            logging.info(f"Regime-Conditioned AR Model successfully trained for regime: {regime}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies predictions dynamically based on the mapped active regime row-by-row."""
        df_lagged = self.prepare_lags(df)
        predictions = np.zeros(len(df_lagged))
        
        # Row-by-row prediction simulation to prevent index leaks
        for idx, (loc_idx, row) in enumerate(df_lagged.iterrows()):
            active_regime = row['market_regime']
            
            X_row = row[self.all_features].values.reshape(1, -1)
            pred_return = self.models[active_regime].predict(X_row)[0]
            predictions[idx] = pred_return
            
        df_lagged['predicted_target_return'] = predictions
        return df_lagged
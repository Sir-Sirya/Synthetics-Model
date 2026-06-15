# main.py
import logging
import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from src.data_loader import MT5DataPipeline, select_target_asset
from src.features import FeatureEngineer
from src.models import MarketRegimeClustering, RegimeConditionedARModel
from src.execution import MT5ExecutionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_live_iteration(pipeline, engineer, clustering_engine, ar_engine, execution_manager):
    """Executes a single real-time calculation and order entry evaluation sequence."""
    if execution_manager.get_open_positions_count() >= 1:
        logging.info("Active position detected. Standing down to guard equity exposure limits.")
        return

    # Fetch the live tracking block
    raw_data = pipeline.build_synchronized_dataset(30)
    
    # Generate structural and Fibonacci features
    featured_data = engineer.generate_market_features(raw_data)
    
    # Map GMM clusters across the active rolling slice
    X_all = featured_data[clustering_engine.feature_cols].values
    X_all_scaled = clustering_engine.scaler.transform(X_all)
    all_clusters = clustering_engine.gmm.predict(X_all_scaled)
    featured_data['market_regime'] = pd.Series(all_clusters, index=featured_data.index).map(clustering_engine.regime_map)

    latest_row = featured_data.iloc[[-1]]
    active_regime = latest_row['market_regime'].values[0]
    logging.info(f"Active Market structural condition identified as: {active_regime}")
    
    # Run autoregressive predictions
    lagged_data = ar_engine.prepare_lags(featured_data)
    predicted_df = ar_engine.predict(lagged_data)
    latest_prediction = predicted_df['predicted_target_return'].iloc[-1]
    
    atr_val = float(latest_row['atr'].iloc[-1])
    account_info = mt5.account_info()
    if account_info is None:
        logging.warning("Temporary inability to fetch MT5 account metrics data.")
        return
        
    balance = account_info.balance
    tick_info = mt5.symbol_info_tick(pipeline.symbol_str)
    if tick_info is None:
        logging.warning("Tick feed dropped momentarily. Skipping this polling loop iteration.")
        return
    
    sl_distance_points = atr_val * 2.5
    tp_distance_points = atr_val * 4.0
    volume_lots = execution_manager.calculate_risk_lot_size(balance, risk_pct=2.0, stop_loss_points=sl_distance_points)
    
    # Signal parsing logic
    if latest_prediction > (atr_val * 0.1) and active_regime in ["Expansion", "Retracement"]:
        if float(latest_row['feat_pd_roll_3d'].iloc[-1]) < 0.5:
            logging.info(f"Bullish Edge Confirmed. Forecast: {latest_prediction:.2f} | Sending BUY Payload...")
            execution_manager.execute_market_order(
                order_type=mt5.ORDER_TYPE_BUY, volume=volume_lots, price=tick_info.ask,
                sl=tick_info.ask - sl_distance_points, tp=tick_info.ask + tp_distance_points
            )
            
    elif latest_prediction < -(atr_val * 0.1) and active_regime in ["Expansion", "Reversal"]:
        if float(latest_row['feat_pd_roll_3d'].iloc[-1]) > 0.5:
            logging.info(f"Bearish Edge Confirmed. Forecast: {latest_prediction:.2f} | Sending SELL Payload...")
            execution_manager.execute_market_order(
                order_type=mt5.ORDER_TYPE_SELL, volume=volume_lots, price=tick_info.bid,
                sl=tick_info.bid + sl_distance_points, tp=tick_info.bid - tp_distance_points
            )
    else:
        logging.info(f"No execution edge present. Forecast momentum variance threshold flat: {latest_prediction:.2f}")

if __name__ == "__main__":
    selected_asset = select_target_asset()
    try:
        pipeline = MT5DataPipeline(selected_asset, mt5.TIMEFRAME_M15)
        engineer = FeatureEngineer()
        clustering_engine = MarketRegimeClustering(n_regimes=4)
        ar_engine = RegimeConditionedARModel(alpha=1.0)
        execution_manager = MT5ExecutionManager(pipeline.symbol_str)
        
        logging.info("Optimizing baseline structural historical parameters (30,000 bars)...")
        historical_raw = pipeline.build_synchronized_dataset(30000)
        historical_featured = engineer.generate_market_features(historical_raw)
        
        regime_matrix = clustering_engine.fit_predict(historical_featured)
        ar_matrix = ar_engine.prepare_lags(regime_matrix)
        ar_engine.fit(ar_matrix)
        logging.info("System optimization complete. Entering fault-tolerant live monitoring loop.")
        
        # --- RESILIENT RUNTIME EXECUTION LOOP ---
        while True:
            try:
                process_live_iteration(pipeline, engineer, clustering_engine, ar_engine, execution_manager)
            except Exception as loop_error:
                # Catching errors INSIDE the loop ensures connection glitches don't kill the script
                logging.warning(f"Data stream or connection glitch intercepted: {str(loop_error)}. Retrying in 10s...")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        logging.info("Production interface loop halted via user command prompt sequence.")
    except Exception as critical_error:
        logging.error(f"Fatal initialization panic: {str(critical_error)}")
    finally:
        mt5.shutdown()
        logging.info("Core engine disconnected clean from terminal database context.")
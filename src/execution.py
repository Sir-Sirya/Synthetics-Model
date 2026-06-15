import logging
import MetaTrader5 as mt5
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MT5ExecutionManager:
    def __init__(self, symbol_str: str, magic_number: int = 202606):
        """
        Handles position tracking, risk-adjusted position sizing, 
        and order execution routines via the native MT5 interface.
        """
        self.symbol_str = symbol_str
        self.magic_number = magic_number

    def get_open_positions_count(self) -> int:
        """Queries active positions tied exclusively to this bot's magic number."""
        positions = mt5.positions_get(symbol=self.symbol_str)
        if positions is None:
            return 0
        bot_positions = [p for p in positions if p.magic == self.magic_number]
        return len(bot_positions)

    def calculate_risk_lot_size(self, account_balance: float, risk_pct: float, stop_loss_points: float) -> float:
        """
        Enforces a strict risk model (Max 2% total equity risk per trade).
        Dynamically adjusts contract volume based on the structural stop distance.
        """
        if stop_loss_points <= 0:
            return 0.01  # Fallback to absolute minimum contract size
            
        symbol_info = mt5.symbol_info(self.symbol_str)
        if symbol_info is None:
            raise ValueError(f"Failed to fetch symbol metrics for {self.symbol_str}")
            
        # Determine raw cash value exposed to risk
        cash_risk = account_balance * (risk_pct / 100.0)
        
        # Calculate standard point value assignment for sizing conversion
        point_value = symbol_info.trade_tick_value / (symbol_info.trade_tick_size + 1e-8)
        calculated_lots = cash_risk / (stop_loss_points * point_value + 1e-8)
        
        # Normalization constraints against broker volume floors/ceilings
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        step_lot = symbol_info.volume_step
        
        normalized_lots = round(calculated_lots / step_lot) * step_lot
        return float(np.clip(normalized_lots, min_lot, max_lot))

    def execute_market_order(self, order_type: int, volume: float, price: float, sl: float, tp: float):
        """Constructs and transmits trade requests directly to the MT5 server."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol_str,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": self.magic_number,
            "comment": "SMC AR-Regime Bot Execution",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None:
            logging.error("Execution failed: Terminal interface returned a null tracking handle.")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order rejected by execution server. Code: {result.retcode} | Info: {result.comment}")
            return False
            
        logging.info(f"Order successfully filled on {self.symbol_str}! Ticket: {result.deal} | Lots: {volume}")
        return True
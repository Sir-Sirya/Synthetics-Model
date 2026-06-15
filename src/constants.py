from enum import Enum

class SyntheticSymbols(Enum):
    # Standard Volatility Indices (Ticks every 2 seconds)
    V10 = "Volatility 10 Index"
    V25 = "Volatility 25 Index"
    V50 = "Volatility 50 Index"
    V75 = "Volatility 75 Index"
    V100 = "Volatility 100 Index"
    
    # (1s) Volatility Indices (Ticks every 1 second)
    V10_1S = "Volatility 10 (1s) Index"
    V15_1S = "Volatility 15 (1s) Index"
    V25_1S = "Volatility 25 (1s) Index"
    V30_1S = "Volatility 30 (1s) Index"
    V50_1S = "Volatility 50 (1s) Index"
    V75_1S = "Volatility 75 (1s) Index"
    V90_1S = "Volatility 90 (1s) Index"
    V100_1S = "Volatility 100 (1s) Index"
    V150_1S = "Volatility 150 (1s) Index"
    V250_1S = "Volatility 250 (1s) Index"  # Fixed typo from image string "250 (1s)"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_
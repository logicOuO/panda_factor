import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class StrFactor(BaseCustomFactor):
    """日内强度因子（STR）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        amount = self.get_series(factors, "amount")

        str_c = (close - low) / (high - low + 1e-4)
        str_b = self.ABS(close - open_) / (high - low + 1e-4)
        str_r = (close - open_) / (open_ + 1e-4)

        sum_c_amt = self.SUM(close * amount, 10)
        sum_amt = self.SUM(amount, 10)
        vwap = sum_c_amt / (sum_amt + 1e-4)
        str_v = (close - vwap) / (close + 1e-4)

        return pd.DataFrame({
            "str_c": str_c,
            "str_b": str_b,
            "str_r": str_r,
            "str_v": str_v
        })

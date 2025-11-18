import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class MnFactor(BaseCustomFactor):
    """微观噪声因子（MN）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        amount = self.get_series(factors, "amount")

        diff_co = self.ABS(close - open_)
        mn_n = pd.Series(
            np.where(diff_co > 0, np.minimum((high - low) / (diff_co + 1e-4) - 1, 50), 50),
            index=close.index
        )

        sum_c_amt = self.SUM(close * amount, 10)
        sum_amt = self.SUM(amount, 10)
        vwap = sum_c_amt / (sum_amt + 1e-4)
        mn_v = self.ABS(close - vwap) / (close + 1e-4)

        mn_o = self.ABS(open_ - self.REF(close, 1)) / (high - low + 1e-4)

        mn_c = (close - open_) / (open_ + 1e-4) - (close - vwap) / (close + 1e-4)

        return pd.DataFrame({
            "mn_n": mn_n,
            "mn_v": mn_v,
            "mn_o": mn_o,
            "mn_c": mn_c
        })

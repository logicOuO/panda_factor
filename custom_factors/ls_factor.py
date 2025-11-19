import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class LSFactor(BaseCustomFactor):
    """流动性突变因子（LS）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        amount = self.get_series(factors, "amount")
        volume = self.get_series(factors, "vol")
        capital = self.get_series(factors, "capital")

        range_ratio = (high - low) / (close + 1e-4)
        amt_z = (amount - self.MA(amount, 20)) / (self.STDDEV(amount, 20) + 1e-4)
        range_z = (range_ratio - self.MA(range_ratio, 20)) / (self.STDDEV(range_ratio, 20) + 1e-4)
        ls_s = amt_z - range_z

        amt_per_vol = amount / (volume + 1e-4)
        ls_b = (amt_per_vol - self.REF(amt_per_vol, 1)) / (self.REF(amt_per_vol, 1) + 1e-4)

        ret = close / (self.REF(close, 1) + 1e-12) - 1
        illiq_a = self.ABS(ret) / (amount / (capital * close + 1e-4) + 1e-4)
        ls_i = (illiq_a > (self.MA(illiq_a, 20) + 2 * self.STDDEV(illiq_a, 20))).astype(float)

        turn1 = volume / (capital + 1e-4)
        ls_t = (turn1 > (self.MA(turn1, 20) + 2 * self.STDDEV(turn1, 20))).astype(float)

        return pd.DataFrame({
            "ls_s": ls_s,
            "ls_b": ls_b,
            "ls_i": ls_i,
            "ls_t": ls_t
        })

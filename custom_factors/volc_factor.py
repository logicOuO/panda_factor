import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class VolCFactor(BaseCustomFactor):
    """波动率收缩因子（VOLC）"""

    def calculate(self, factors):
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        close = self.get_series(factors, "close")

        prev_close = self.REF(close, 1)
        hl_range = high - low
        tr1 = self.MAX(hl_range, self.ABS(high - prev_close))
        tr1 = self.MAX(tr1, self.ABS(low - prev_close))
        atr10 = self.MA(tr1, 10)
        volc_a = atr10 / (self.HHV(atr10, 20) + 1e-4)

        ret = close / (prev_close + 1e-12) - 1
        std_ret5 = self.STD(ret, 5)
        volc_s = std_ret5 / (self.HHV(std_ret5, 15) + 1e-4)

        range_ratio = hl_range / (close + 1e-4)
        volc_r = range_ratio / (self.HHV(range_ratio, 20) + 1e-4)

        volc_b = (self.HHV(close, 20) - self.LLV(close, 20)) / (self.MA(close, 20) + 1e-4)

        return pd.DataFrame({
            "volc_a": volc_a,
            "volc_s": volc_s,
            "volc_r": volc_r,
            "volc_b": volc_b
        })

import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class GapFactor(BaseCustomFactor):
    """缺口因子（GAP）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")

        prev_high = self.REF(high, 1)
        prev_low = self.REF(low, 1)
        prev_close = self.REF(close, 1)

        overlap = ((prev_high > low) & (prev_low < high))
        gap_i = (~overlap).astype(float)
        gap_s = (open_ - prev_close) / (prev_close + 1e-4)
        gap_f = (close - open_) / (open_ - prev_close + 1e-4)
        gap_fo = open_ / (prev_close + 1e-4) - 1

        return pd.DataFrame({
            "gap_i": gap_i,
            "gap_s": gap_s,
            "gap_f": gap_f,
            "gap_fo": gap_fo
        })

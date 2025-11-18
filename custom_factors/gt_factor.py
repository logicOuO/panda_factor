import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class GTFactor(BaseCustomFactor):
    """博弈强弱因子（GT）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")

        gt_u = (high - self.MAX(open_, close)) / (high - low + 1e-4)
        gt_l = (self.MIN(open_, close) - low) / (high - low + 1e-4)

        prev_close = self.REF(close, 1)
        prev_open_close = self.REF(close - open_, 1)
        prev2_close = self.REF(close, 2)
        gt_e = self.SIGN(close - open_) * self.ABS(close - open_) / (prev_close + 1e-4)
        denom = self.ABS(prev_open_close) / (prev2_close + 1e-4) + 1e-4
        gt_e = gt_e / denom
        amp_ratio = self.ABS(close - open_) / (prev_close + 1e-4)
        amp_prev = 1.5 * self.ABS(prev_open_close) / (prev2_close + 1e-4)
        gt_e = gt_e * (amp_ratio > amp_prev).astype(float)

        gt_d = (self.ABS(close - open_) / (high - low + 1e-4) < 0.1).astype(float)

        return pd.DataFrame({
            "gt_u": gt_u,
            "gt_l": gt_l,
            "gt_e": gt_e,
            "gt_d": gt_d
        })

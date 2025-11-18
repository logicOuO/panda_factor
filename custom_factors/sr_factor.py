import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class SRFactor(BaseCustomFactor):
    """支撑阻力因子（SR）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")

        sr_b = close > self.HHV(high, 20)
        prev_close = self.REF(close, 1)
        sr_g = (open_ - prev_close) / (prev_close + 1e-4)
        sr_p = (close - self.LLV(low, 10)) / (self.HHV(high, 10) - self.LLV(low, 10) + 1e-4)
        sr_ps = self.MA((close > open_).astype(float), 12)

        return pd.DataFrame({
            "sr_b": sr_b.astype(float),
            "sr_g": sr_g,
            "sr_p": sr_p,
            "sr_ps": sr_ps
        })

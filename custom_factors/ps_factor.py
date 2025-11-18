import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class PSFactor(BaseCustomFactor):
    """价格结构因子（PS）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")

        ps_i = ((high <= self.REF(high, 1)) & (low >= self.REF(low, 1))).astype(float)
        ps_id = ps_i * self.SIGN(close - self.REF(close, 1))

        ps_g = close - (self.HHV(close, 3) + self.LLV(close, 3)) / 2.0

        ps_f = (
            (high == self.HHV(high, 5)) &
            (close < open_) &
            ((self.HHV(close, 5) - self.LLV(close, 5)) / (self.MA(close, 5) + 1e-4) < 0.05)
        ).astype(float)

        return pd.DataFrame({
            "ps_i": ps_i,
            "ps_id": ps_id,
            "ps_g": ps_g,
            "ps_f": ps_f
        })

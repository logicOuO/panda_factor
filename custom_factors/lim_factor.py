import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class LimFactor(BaseCustomFactor):
    """涨跌停因子（LIM）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")

        prev_close = self.REF(close, 1)
        limit_up = (prev_close * 1.1).round(2)
        limit_down = (prev_close * 0.9).round(2)

        lim_u = (close >= limit_up * 0.998).astype(float)
        lim_d = (close <= limit_down * 1.002).astype(float)

        limit_up_prev = self.REF(limit_up, 1)
        lim_c = (
            (high >= limit_up * 0.998) &
            (close < limit_up * 0.998) &
            (prev_close < limit_up_prev * 0.998)
        ).astype(float)

        lim_o = (
            (self.ABS(open_ - close) / (close + 1e-4) < 0.001) &
            (close >= limit_up * 0.998)
        ).astype(float)

        return pd.DataFrame({
            "lim_u": lim_u,
            "lim_d": lim_d,
            "lim_c": lim_c,
            "lim_o": lim_o
        })

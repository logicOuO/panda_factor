import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class SkewFactor(BaseCustomFactor):
    """偏度因子（SKEW）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")

        ret = close / (self.REF(close, 1) + 1e-12) - 1
        ma_ret = self.MA(ret, 20)
        std_ret = self.STDDEV(ret, 20)
        norm_ret = (ret - ma_ret) / (std_ret + 1e-4)
        skew_r = self.MA(np.power(norm_ret, 3), 20)

        skew_c = self.EMA(np.power(ret - ma_ret, 3), 20) / (np.power(std_ret, 3) + 1e-4)

        co_ratio = (close - open_) / (open_ + 1e-4)
        co_ma = self.MA(co_ratio, 20)
        skew_t = self.MA(np.power(co_ratio - co_ma, 3), 20)

        hl_ratio = high / (low + 1e-4)
        hl_ma = self.MA(hl_ratio, 20)
        skew_h = self.MA(np.power(hl_ratio - hl_ma, 3), 20)

        return pd.DataFrame({
            "skew_r": skew_r,
            "skew_c": skew_c,
            "skew_t": skew_t,
            "skew_h": skew_h
        })

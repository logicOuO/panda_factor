import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class NlFactor(BaseCustomFactor):
    """非线性因子（NL）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")

        nl_h0 = self.SLOPE(close, 15) / (self.STDDEV(close, 15) + 1e-4)
        nl_l0 = self.ABS(close - self.REF(close, 3)) / (self.STDDEV(close, 10) + 1e-4)
        ma5 = self.MA(close, 5)
        ma20 = self.MA(close, 20)
        nl_q0 = np.power(ma5 - ma20, 2) / (np.power(self.STDDEV(close, 20), 2) + 1e-4)
        nl_s0 = 1 / (1 + np.exp(-10 * (close - self.MA(close, 10)) / (self.STDDEV(close, 10) + 1e-4)))

        return pd.DataFrame({
            "nl_h0": nl_h0,
            "nl_l0": nl_l0,
            "nl_q0": nl_q0,
            "nl_s0": nl_s0
        })

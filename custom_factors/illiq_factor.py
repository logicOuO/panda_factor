import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class IlliqFactor(BaseCustomFactor):
    """非流动性因子（ILLIQ）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        amount = self.get_series(factors, "amount")
        volume = self.get_series(factors, "vol")
        capital = self.get_series(factors, "capital")

        ret = np.log(close / (self.REF(close, 1) + 1e-12))
        illiq_a = self.ABS(ret) / (amount / (capital * close + 1e-4) + 1e-4)
        illiq_t = 1 / (volume / (capital + 1e-4) + 1e-4)
        illiq_z = self.MA((ret == 0).astype(float), 10)
        illiq_r = (high - low) / (amount + 1e-4) * 1_000_000

        return pd.DataFrame({
            "illiq_ret": ret,
            "illiq_a": illiq_a,
            "illiq_t": illiq_t,
            "illiq_z": illiq_z,
            "illiq_r": illiq_r
        })

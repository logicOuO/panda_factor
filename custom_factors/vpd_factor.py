import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class VPDFactor(BaseCustomFactor):
    """量价背离因子（VPD）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        amount = self.get_series(factors, "amount")
        volume = self.get_series(factors, "vol")

        ret = close / (self.REF(close, 1) + 1e-12) - 1
        ret_ma = self.MA(ret, 20)
        ret_std = self.STD(ret, 20)

        amt_ma = self.MA(amount, 20)
        amt_std = self.STD(amount, 20)

        vpd_z1 = (ret - ret_ma) / (ret_std + 1e-4) - (amount - amt_ma) / (amt_std + 1e-4)

        obv = self.OBV(close, volume)
        sign_obv = self.SIGN(obv - self.REF(obv, 5))
        sign_price = self.SIGN(close - self.REF(close, 5))
        vpd_o = (sign_obv != sign_price).astype(float)

        vpd_h = ((ret < 0) & (amount > self.REF(amount, 5))).astype(float)

        co_ratio = self.ABS(close - open_) / (self.REF(close, 1) + 1e-4)
        avg_co = self.MA(co_ratio, 5)
        vpd_b = ((co_ratio < avg_co) & (amount > self.REF(amount, 5))).astype(float)

        return pd.DataFrame({
            "vpd_z1": vpd_z1,
            "vpd_o": vpd_o,
            "vpd_h": vpd_h,
            "vpd_b": vpd_b
        })

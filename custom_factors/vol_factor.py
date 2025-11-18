import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class VolFactor(BaseCustomFactor):
    """波动因子（VOL）"""

    def calculate(self, factors):
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        close = self.get_series(factors, "close")

        prev_close = self.REF(close, 1)
        hl_range = high - low
        tr1 = self.MAX(hl_range, self.ABS(high - prev_close))
        tr1 = self.MAX(tr1, self.ABS(low - prev_close))
        atr14 = self.MA(tr1, 14)
        atr10 = self.MA(tr1, 10)
        atr20_h = self.HHV(atr10, 20)
        vol_c = 1 - atr10 / (atr20_h + 1e-4)
        vol_hl = hl_range / (close + 1e-4)

        return pd.DataFrame({
            "vol_tr1": tr1,
            "vol_atr14": atr14,
            "vol_atr10": atr10,
            "vol_atr20_h": atr20_h,
            "vol_vol_c": vol_c,
            "vol_vol_hl": vol_hl
        })

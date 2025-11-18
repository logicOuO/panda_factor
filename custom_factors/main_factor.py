import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class MainFactor(BaseCustomFactor):
    """主力行为因子（MAIN）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        amount = self.get_series(factors, "amount")
        capital = self.get_series(factors, "capital")

        med_a = self.MA(amount, 20)
        med_r = self.MA(high / (low + 1e-4) - 1, 20)
        sqz = (amount > 2 * med_a) & ((high / (low + 1e-4) - 1) < 2 * med_r)

        sum_c_amt = self.SUM(close * amount, 5)
        sum_amt = self.SUM(amount, 5)
        vw5 = sum_c_amt / (sum_amt + 1e-4)
        main_v = close > vw5 * 1.02

        chp = amount / (capital + 1e-4)

        return pd.DataFrame({
            "main_med_a": med_a,
            "main_med_r": med_r,
            "main_sqz": sqz.astype(float),
            "main_vw5": vw5,
            "main_main_v": main_v.astype(float),
            "main_chp": chp
        })

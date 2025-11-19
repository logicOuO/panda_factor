import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class VSFactor(BaseCustomFactor):
    """成交量结构因子（VS）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")
        amount = self.get_series(factors, "amount")

        sum_c_amt = self.SUM(close * amount, 20)
        sum_amt = self.SUM(amount, 20)
        vs_v = 3 * (sum_c_amt / (sum_amt + 1e-4) - self.MA(close, 20)) / (self.STDDEV(close, 20) + 1e-4)

        body_balance = ((close - low) - (high - close)) / (high - low + 1e-4)
        vs_c = self.SUM(body_balance * amount, 20)

        vs_p = amount / (sum_amt + 1e-4)

        return pd.DataFrame({
            "vs_v": vs_v,
            "vs_c": vs_c,
            "vs_p": vs_p,
            "vs_vp": vs_v
        })

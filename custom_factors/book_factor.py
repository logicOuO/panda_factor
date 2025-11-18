import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class BookFactor(BaseCustomFactor):
    """盘口代理因子（BOOK）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        open_ = self.get_series(factors, "open")
        high = self.get_series(factors, "high")
        low = self.get_series(factors, "low")

        cpr = (close - low) / (high - low + 1e-4)
        body = self.ABS(close - open_) / (high - low + 1e-4)
        us1 = (high - self.MAX(open_, close)) / (high - low + 1e-4)
        gap_m = open_ - (self.REF(high, 1) + self.REF(low, 1)) / 2.0

        return pd.DataFrame({
            "book_cpr": cpr,
            "book_body": body,
            "book_us1": us1,
            "book_gap_m": gap_m
        })

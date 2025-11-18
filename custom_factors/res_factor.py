import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class ResFactor(BaseCustomFactor):
    """相对强度因子（RES）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        idx_series = None
        for key in ["idx_close", "index_close", "benchmark_close", "idx1", "999999_close"]:
            try:
                idx_series = self.get_series(factors, key)
                break
            except KeyError:
                continue
        if idx_series is None:
            raise KeyError("ResFactor 需要指数收盘价列，如 idx_close/index_close")

        c5 = close / (self.REF(close, 5) + 1e-12) - 1
        idx5 = idx_series / (self.REF(idx_series, 5) + 1e-12) - 1
        res_5 = c5 - idx5
        res_b = self.SLOPE(close, 20)
        idx_ret = idx_series / (self.REF(idx_series, 1) + 1e-12) - 1
        ret = close / (self.REF(close, 1) + 1e-12) - 1
        res_a = self.MA(ret - res_b * idx_ret, 10)

        return pd.DataFrame({
            "res_5": res_5,
            "res_b": res_b,
            "res_a": res_a,
            "res_r": c5
        })

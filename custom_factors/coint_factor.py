import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class CointFactor(BaseCustomFactor):
    """协整因子（COINT）"""

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
            raise KeyError("CointFactor 需要指数收盘价列，如 idx_close/index_close")

        beta_v = self.SLOPE(close, 20)
        res1 = idx_series - beta_v * close
        res_ma = self.MA(res1, 20)
        res_std = self.STD(res1, 20)
        res_z1 = (res1 - res_ma) / (res_std + 1e-4)

        return pd.DataFrame({
            "coint_idx": idx_series,
            "coint_beta": beta_v,
            "coint_res1": res1,
            "coint_res_z1": res_z1
        })

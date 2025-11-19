import numpy as np
import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class TailFactor(BaseCustomFactor):
    """尾部风险因子（TAIL）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        ret = np.log(close / (self.REF(close, 1) + 1e-12))
        tail_md = self.LLV(ret, 20)
        tail_sk = self.STDDEV(ret, 20)
        tail_v1 = self.EMA(ret, 10)
        tail_v = self.LLV(tail_v1, 40)
        abs_ret = self.ABS(ret)
        tail_j = abs_ret > self.HHV(abs_ret, 250) * 0.95

        return pd.DataFrame({
            "tail_ret": ret,
            "tail_md": tail_md,
            "tail_sk": tail_sk,
            "tail_v1": tail_v1,
            "tail_v": tail_v,
            "tail_j": tail_j.astype(float)
        })

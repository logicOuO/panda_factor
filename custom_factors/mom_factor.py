"""
使用项目内 Factor/FactorUtils 计算 factor.md 中“动量 MOM”因子。
输出列：mom_q, mom_m20, mom_m5, mom_m1
"""
from panda_factor.generate.factor_base import Factor

class MomFactor(Factor):
    """动量因子：M20/M5/M1 与 MOM_Q"""

    def calculate(self, factors):
        # FactorDataWrapper 会返回 FactorSeries，这里取 .series 以防直接给 pd.Series
        close = factors["close"].series if hasattr(factors["close"], "series") else factors["close"]

        # 按单股票序列进行简单位移，用 REF 与通达信一致（不做分组）
        m20 = close / self.REF(close, 20) - 1
        m5 = close / self.REF(close, 5) - 1
        m1 = close / self.REF(close, 1) - 1

        ma20 = self.MA(close, 20)
        std20 = self.STD(close, 20)
        diff = close - ma20
        mom_q = self.SIGN(diff) * (diff ** 2) / (std20 ** 2 + 1e-4)

        out = mom_q.to_frame("mom_q")
        out["mom_m20"] = m20
        out["mom_m5"] = m5
        out["mom_m1"] = m1
        return out

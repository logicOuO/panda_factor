"""
自定义资金流因子（对应 factor.md 中 FLOW 段落）。

公式：
    AMT = AMOUNT
    MCAP = FINANCE(1) * C   （即流通股本 * 收盘价，若直接提供市值字段则优先使用）
    FLOW_MV1 = AMT / (MCAP + 1e-4)
    FLOW_MV = MIN(MAX(FLOW_MV1, 0.01), 100)
"""
from custom_factors.base_factor import BaseCustomFactor


class FlowFactor(BaseCustomFactor):
    """资金流因子"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        amount = self.get_series(factors, "amount")
        try:
            market_cap = self.get_series(factors, "market_cap")
        except KeyError:
            capital = self.get_series(factors, "capital")
            market_cap = capital * close

        flow_mv1 = amount / (market_cap + 1e-4)
        flow_mv = self.MIN(self.MAX(flow_mv1, 0.01), 100.0)
        return flow_mv.to_frame("flow_mv")

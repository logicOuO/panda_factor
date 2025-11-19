import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class TurnFactor(BaseCustomFactor):
    """换手率异常因子（TURN）"""

    def calculate(self, factors):
        volume = self.get_series(factors, "vol")
        capital = self.get_series(factors, "capital")

        turn1 = volume / (capital + 1e-4)
        turn_z = (turn1 - self.MA(turn1, 20)) / (self.STDDEV(turn1, 20) + 1e-4)
        turn_q = turn1 / (self.HHV(turn1, 250) + 1e-4) * 0.9
        turn_j = (turn1 / (self.REF(turn1, 1) + 1e-4) > 2).astype(float)
        turn_c = (turn1 > 0.3).astype(float)

        return pd.DataFrame({
            "turn_turn1": turn1,
            "turn_z": turn_z,
            "turn_q": turn_q,
            "turn_j": turn_j,
            "turn_c": turn_c
        })

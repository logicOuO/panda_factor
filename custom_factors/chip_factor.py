import pandas as pd

from custom_factors.base_factor import BaseCustomFactor


class ChipFactor(BaseCustomFactor):
    """筹码松动因子（CHIP）"""

    def calculate(self, factors):
        close = self.get_series(factors, "close")
        amount = self.get_series(factors, "amount")
        capital = self.get_series(factors, "capital")
        volume = self.get_series(factors, "vol")

        chip_a = amount / (capital + 1e-4)
        chip_cv = self.STD(chip_a, 10) / (self.MA(chip_a, 10) + 1e-4)
        chip_r = (capital - self.REF(capital, 20)) / (self.REF(capital, 20) + 1e-4)

        try:
            finance46 = self.get_first_available(factors, ["finance46", "finance_46"])
        except KeyError:
            raise KeyError("ChipFactor 需要 finance46/finance_46 字段")
        turn1 = volume / (capital + 1e-4)
        chip_f = finance46 / (capital + 1e-4) - turn1

        return pd.DataFrame({
            "chip_a": chip_a,
            "chip_cv": chip_cv,
            "chip_r": chip_r,
            "chip_f": chip_f
        })

from panda_factor.generate.factor_base import Factor


class BaseCustomFactor(Factor):
    """自定义因子基类，提供常用Series解析与字段获取"""

    @staticmethod
    def _to_series(value):
        return value.series if hasattr(value, "series") else value

    def get_series(self, factors, key: str):
        """获取必需字段"""
        return self._to_series(factors[key])

    def get_optional_series(self, factors, key: str, default=None):
        try:
            return self.get_series(factors, key)
        except KeyError:
            if default is not None:
                return default
            raise

    def get_first_available(self, factors, keys, default=None):
        for key in keys:
            try:
                return self.get_series(factors, key)
            except KeyError:
                continue
        if default is not None:
            return default
        joined = "/".join(keys)
        raise KeyError(f"{self.__class__.__name__} 需要字段: {joined}")

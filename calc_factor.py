"""
通用因子批量计算脚本，可选择不同自定义因子（如 mom、flow）。

示例：
    python calc_factor.py --factor mom
    python calc_factor.py --factor flow --data-dir datasets --output-dir output
"""
import argparse
import glob
import os
import sys
from typing import Dict, Iterable, Tuple

import pandas as pd

# 让项目包可导入
ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "panda_factor")):
    if path not in sys.path:
        sys.path.insert(0, path)

from panda_factor.generate.factor_wrapper import FactorDataWrapper  # type: ignore
from custom_factors.mom_factor import MomFactor  # type: ignore
from custom_factors.flow_factor import FlowFactor  # type: ignore

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量计算因子")
    parser.add_argument("--factor", choices=("mom", "flow"), default="mom", help="选择因子名称")
    parser.add_argument("--data-dir", default="datasets", help="日线数据目录")
    parser.add_argument("--output-dir", default="output", help="结果输出根目录（每股单独 CSV）")
    parser.add_argument("--log-step", type=int, default=200, help="每处理多少只股票打印一次进度")
    return parser.parse_args()


def iter_stock_frames(files: Iterable[str]) -> Iterable[Tuple[str, pd.DataFrame]]:
    """
    逐个读取股票 CSV，保持原列名不变，只补充索引信息
    """
    symbol_candidates = ["股票代码", "code"]
    date_candidates = ["date", "datetime", "trade_date", "日期"]

    for path in files:
        df = pd.read_csv(path)
        df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]

        symbol_col = next((c for c in symbol_candidates if c in df.columns), None)
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if symbol_col is None or date_col is None:
            raise KeyError(f"{os.path.basename(path)} 缺少 股票代码 或 date/datetime 列")

        df = df.copy()
        df["_date_internal"] = pd.to_datetime(df[date_col])
        df["_symbol_internal"] = df[symbol_col].astype(str).str.zfill(6)
        df.sort_values("_date_internal", inplace=True)
        df.set_index(["_date_internal", "_symbol_internal"], inplace=True)
        df.index.names = ["date", "code"]
        df.drop(columns=[col for col in ["_date_internal", "_symbol_internal"] if col in df.columns], inplace=True, errors="ignore")
        yield os.path.basename(path), df


def build_factor_wrapper(df: pd.DataFrame) -> FactorDataWrapper:
    factors: Dict[str, pd.Series] = {}
    for col in df.columns:
        factors[col] = df[col]
    return FactorDataWrapper(factors)


def main():
    args = parse_args()
    factor_cls = MomFactor if args.factor == "mom" else FlowFactor

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到数据文件，检查目录: {args.data_dir}")

    factor_out_dir = os.path.join(args.output_dir, args.factor)
    os.makedirs(factor_out_dir, exist_ok=True)

    factor_obj = factor_cls()
    total_rows = 0

    for idx, (fname, df) in enumerate(iter_stock_frames(files), start=1):
        wrapper = build_factor_wrapper(df)
        result_df = factor_obj.calculate(wrapper).reset_index()
        result_df = result_df.astype({col: "float32" for col in result_df.select_dtypes(include=["float64"]).columns})

        code = df.index.get_level_values("code")[0]
        stock_path = os.path.join(factor_out_dir, f"{code}.csv")
        result_df.to_csv(stock_path, index=False)
        total_rows += len(result_df)

        if idx % args.log_step == 0:
            print(f"  -> 已处理 {idx}/{len(files)} 只股票, 累计 {total_rows:,} 行")

    print(f"✅ 完成：{idx} 只股票，总 {total_rows:,} 行，输出目录 {factor_out_dir}")


if __name__ == "__main__":
    main()

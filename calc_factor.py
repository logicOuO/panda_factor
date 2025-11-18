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
import multiprocessing as mp
from typing import Dict, Tuple

import pandas as pd

# 让项目包可导入
ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "panda_factor")):
    if path not in sys.path:
        sys.path.insert(0, path)

from panda_factor.generate.factor_wrapper import FactorDataWrapper  # type: ignore
from custom_factors.mom_factor import MomFactor  # type: ignore
from custom_factors.flow_factor import FlowFactor  # type: ignore
from custom_factors.vol_factor import VolFactor  # type: ignore
from custom_factors.book_factor import BookFactor  # type: ignore
from custom_factors.main_factor import MainFactor  # type: ignore
from custom_factors.tail_factor import TailFactor  # type: ignore
from custom_factors.illiq_factor import IlliqFactor  # type: ignore
from custom_factors.coint_factor import CointFactor  # type: ignore
from custom_factors.skew_factor import SkewFactor  # type: ignore
from custom_factors.sr_factor import SRFactor  # type: ignore
from custom_factors.gap_factor import GapFactor  # type: ignore
from custom_factors.vpd_factor import VPDFactor  # type: ignore
from custom_factors.turn_factor import TurnFactor  # type: ignore
from custom_factors.chip_factor import ChipFactor  # type: ignore
from custom_factors.str_factor import StrFactor  # type: ignore
from custom_factors.volc_factor import VolCFactor  # type: ignore
from custom_factors.res_factor import ResFactor  # type: ignore
from custom_factors.lim_factor import LimFactor  # type: ignore
from custom_factors.ps_factor import PSFactor  # type: ignore
from custom_factors.vs_factor import VSFactor  # type: ignore
from custom_factors.nl_factor import NlFactor  # type: ignore
from custom_factors.mn_factor import MnFactor  # type: ignore
from custom_factors.gt_factor import GTFactor  # type: ignore
from custom_factors.ls_factor import LSFactor  # type: ignore

mp_context = mp.get_context("spawn")

FACTOR_MAP = {
    "mom": MomFactor,
    "flow": FlowFactor,
    "vol": VolFactor,
    "book": BookFactor,
    "main": MainFactor,
    "tail": TailFactor,
    "illiq": IlliqFactor,
    "coint": CointFactor,
    "skew": SkewFactor,
    "sr": SRFactor,
    "gap": GapFactor,
    "vpd": VPDFactor,
    "turn": TurnFactor,
    "chip": ChipFactor,
    "str": StrFactor,
    "volc": VolCFactor,
    "res": ResFactor,
    "lim": LimFactor,
    "ps": PSFactor,
    "vs": VSFactor,
    "nl": NlFactor,
    "mn": MnFactor,
    "gt": GTFactor,
    "ls": LSFactor,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量计算因子")
    parser.add_argument("--factor", choices=FACTOR_MAP.keys(), default="mom", help="选择因子名称")
    parser.add_argument("--data-dir", default="datasets", help="日线数据目录")
    parser.add_argument("--output-dir", default="output", help="结果输出根目录（每股单独 CSV）")
    parser.add_argument("--log-step", type=int, default=500, help="每处理多少只股票打印一次进度")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="并行进程数")
    return parser.parse_args()


def load_stock_frame(path: str) -> pd.DataFrame:
    symbol_candidates = ["symbol", "股票代码", "code"]
    date_candidates = ["date", "datetime", "trade_date", "日期"]

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
    df.index.names = ["date", "symbol"]  # 保持与 FactorUtils 内部 groupby(level='symbol') 一致
    df.drop(columns=[col for col in ["_date_internal", "_symbol_internal"] if col in df.columns], inplace=True, errors="ignore")
    return df


def build_factor_wrapper(df: pd.DataFrame) -> FactorDataWrapper:
    factors: Dict[str, pd.Series] = {}
    for col in df.columns:
        factors[col] = df[col]
    return FactorDataWrapper(factors)


def process_stock_task(task: Tuple[str, str, str]) -> Tuple[str, int]:
    path, factor_name, output_dir = task
    df = load_stock_frame(path)
    factor_obj = FACTOR_MAP[factor_name]()
    wrapper = build_factor_wrapper(df)
    result_df = factor_obj.calculate(wrapper).reset_index()
    float_cols = result_df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        result_df[float_cols] = result_df[float_cols].astype("float32")
    code = df.index.get_level_values("symbol")[0]  # 仍用 symbol 层名取值
    stock_path = os.path.join(output_dir, f"{code}.csv")  # 文件名用 code 形式保存
    result_df.to_csv(stock_path, index=False)
    return code, len(result_df)


def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到数据文件，检查目录: {args.data_dir}")

    factor_out_dir = os.path.join(args.output_dir, args.factor)
    os.makedirs(factor_out_dir, exist_ok=True)

    total_rows = 0

    tasks = [(path, args.factor, factor_out_dir) for path in files]
    with mp_context.Pool(processes=args.workers) as pool:
        for idx, (_, rows) in enumerate(pool.imap_unordered(process_stock_task, tasks), start=1):
            total_rows += rows
            if idx % args.log_step == 0:
                print(f"  -> 已处理 {idx}/{len(files)} 只股票, 累计 {total_rows:,} 行")

    print(f"✅ 完成：{len(files)} 只股票，总 {total_rows:,} 行，输出目录 {factor_out_dir}")


if __name__ == "__main__":
    main()

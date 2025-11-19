"""
只做因子评估（IC / RankIC），不跑回测。

用法示例：
    python eval_factor.py --factor-file-dir output/mom --factor-col mom_q --adjustment-cycle 5 --direction 1

说明：
1. 会合并目录下所有股票因子文件（需包含 date、symbol 或 code，以及指定的因子列）。
2. 内部调用 factor_ic_workflow，通过 panda_data 自动拉行情、做清洗、合并，输出 IC 等指标。
3. 默认使用 spawn 多进程读取因子文件，加快合并。
"""
import argparse
import glob
import os
import sys
import multiprocessing as mp
from typing import List, Tuple

# 降低日志噪音（默认仅错误）
os.environ.setdefault("LOG_LEVEL", "ERROR")

import pandas as pd

# 让项目包可导入
ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRA_PATHS = [
    ROOT,
    os.path.join(ROOT, "panda_factor"),
    os.path.join(ROOT, "panda_common"),
]
for path in EXTRA_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)

mp_context = mp.get_context("spawn")


def parse_args():
    p = argparse.ArgumentParser(description="因子评估（IC / RankIC）")
    target_group = p.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--factor-file-dir", help="单一因子目录（每股一个 CSV）")
    target_group.add_argument(
        "--factor-root",
        help="批量模式：根目录下的每个子目录视为一个因子组，逐个评估",
    )
    p.add_argument("--factor-col", required=False, help="要评估的因子列名；不填则批量评估所有非 date/code 列")
    p.add_argument("--start-date", default=None, help="开始日期，YYYYMMDD 或 YYYY-MM-DD，可选")
    p.add_argument("--end-date", default=None, help="结束日期，YYYYMMDD 或 YYYY-MM-DD，可选")
    p.add_argument("--adjustment-cycle", type=int, default=8, help="调仓周期 / IC 滞后步长")
    p.add_argument("--group-number", type=int, default=10, help="分组数（用于 IC 分组统计，可保留默认）")
    p.add_argument("--direction", type=int, choices=(0, 1), default=1, help="因子方向：1 正向，0 反向")
    p.add_argument("--workers", type=int, default=mp.cpu_count(), help="并行进程数（用于读文件合并）")
    p.add_argument("--mode", choices=("mongo", "local"), default="local", help="mongo 使用 factor_ic_workflow；local 直接用本地行情算 IC")
    p.add_argument("--price-dir", default="datasets", help="mode=local 时，日线行情目录（与 calc_factor 产出的同源数据）")
    p.add_argument(
        "--summary-csv",
        default="output/factor_ic_summary.csv",
        help="汇总结果输出路径，留空表示不写入",
    )
    return p.parse_args()


def _discover_factor_dirs(args) -> List[str]:
    if args.factor_root:
        candidates: List[str] = []
        for entry in sorted(os.listdir(args.factor_root)):
            full_path = os.path.join(args.factor_root, entry)
            if os.path.isdir(full_path):
                candidates.append(full_path)
        if not candidates:
            raise FileNotFoundError(f"未在 {args.factor_root} 下找到有效子目录")
        return candidates
    return [args.factor_file_dir]


def _evaluate_directory(dir_path: str, args) -> pd.DataFrame | None:
    files = glob.glob(os.path.join(dir_path, "*.csv"))
    if not files:
        print(f"[warn] 目录为空: {dir_path}")
        return None

    with mp_context.Pool(processes=args.workers) as pool:
        df_list: List[pd.DataFrame] = pool.starmap(load_one, [(p, args.factor_col) for p in files])

    df_all = pd.concat(df_list, ignore_index=True)
    if args.start_date:
        df_all = df_all[df_all["date"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df_all = df_all[df_all["date"] <= pd.to_datetime(args.end_date)]

    df_all = df_all.sort_values(["date", "symbol"]).reset_index(drop=True)

    if args.factor_col:
        factor_cols = [args.factor_col]
    else:
        exclude_cols = {"date", "symbol", "code", "name"}
        factor_cols = [col for col in df_list[0].columns if col not in exclude_cols]
        if not factor_cols:
            raise ValueError("未检测到可用因子列，至少需要 1 列")

    results = []
    for col in factor_cols:
        df_factor = df_all[["date", "symbol", col]].rename(columns={col: "factor"})

        if args.mode == "mongo":
            from panda_factor.analysis.factor_ic_workflow import factor_ic_workflow  # type: ignore

            factor_ic_workflow(
                df_factor=df_factor,
                adjustment_cycle=args.adjustment_cycle,
                group_number=args.group_number,
                factor_direction=args.direction,
            )
            continue

        summary = compute_local_ic(
            df_factor=df_factor,
            price_dir=args.price_dir,
            lag=args.adjustment_cycle,
        )
        summary.insert(0, "factor", col)
        results.append(summary)

    if results:
        dir_result = pd.concat(results, ignore_index=True)
        print("==== 因子批量评估结果 ====")
        print(dir_result)
        return dir_result
    return None


def load_one(path: str, factor_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    # 识别 symbol / code
    if "symbol" in df.columns:
        sym_col = "symbol"
    elif "code" in df.columns:
        sym_col = "code"
    elif "股票代码" in df.columns:
        sym_col = "股票代码"
    else:
        raise KeyError(f"{os.path.basename(path)} 缺少 symbol/code/股票代码 列")

    # 识别 date
    date_col = None
    for cand in ["date", "datetime", "trade_date", "日期"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise KeyError(f"{os.path.basename(path)} 缺少 date/datetime 列")

    df = df.rename(columns={date_col: "date", sym_col: "symbol"})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df


def load_price_file(args_tuple: Tuple[str, set, int]) -> pd.DataFrame | None:
    path, symbols, lag = args_tuple
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    if "股票代码" not in df.columns or "datetime" not in df.columns or "close" not in df.columns:
        return None
    code = str(df["股票代码"].iloc[0]).zfill(6)
    if code not in symbols:
        return None
    df = df[["datetime", "close", "股票代码"]].rename(columns={"datetime": "date", "股票代码": "symbol"})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = code
    df = df.sort_values("date").reset_index(drop=True)
    df["future_ret"] = df["close"].shift(-lag) / df["close"] - 1
    return df[["date", "symbol", "future_ret"]]


def compute_local_ic(df_factor: pd.DataFrame, price_dir: str, lag: int) -> pd.DataFrame:
    """本地行情计算 IC：读取 price_dir 下 csv，算未来 lag 日收益"""
    price_files = glob.glob(os.path.join(price_dir, "*.csv"))
    if not price_files:
        raise FileNotFoundError(f"行情目录为空: {price_dir}")

    # 只加载需要的 symbol
    symbols = set(df_factor["symbol"].unique())
    tasks = [(path, symbols, lag) for path in price_files]

    with mp_context.Pool(processes=mp.cpu_count()) as pool:
        price_dfs = [df for df in pool.map(load_price_file, tasks) if df is not None]

    price_all = pd.concat(price_dfs, ignore_index=True)
    merged = pd.merge(df_factor, price_all, on=["date", "symbol"], how="inner")
    merged = merged.dropna(subset=["factor", "future_ret"])

    ic_by_date = merged.groupby("date", group_keys=False)[["factor", "future_ret"]].apply(
        lambda df: df["factor"].corr(df["future_ret"], method="spearman")
    ).dropna()

    summary = pd.DataFrame({
        "IC_mean": ic_by_date.mean(),
        "IC_std": ic_by_date.std(),
        "IC_IR": ic_by_date.mean() / (ic_by_date.std() + 1e-6),
        "Positive_IC_ratio": (ic_by_date > 0).mean(),
        "IC_count": len(ic_by_date)
    }, index=[0])

    print("==== Local 模式 IC 统计 ====")
    print(summary)
    return summary


def main():
    args = parse_args()
    factor_dirs = _discover_factor_dirs(args)

    aggregate: List[pd.DataFrame] = []
    for dir_path in factor_dirs:
        group_name = os.path.basename(os.path.normpath(dir_path))
        print(f"\n==== 开始评估因子组: {group_name} ({dir_path}) ====")
        dir_result = _evaluate_directory(dir_path, args)
        if dir_result is None:
            continue
        dir_result.insert(0, "factor_group", group_name)
        aggregate.append(dir_result)

    if aggregate:
        final_df = pd.concat(aggregate, ignore_index=True)
        if len(aggregate) > 1:
            print("\n==== 全部因子组汇总 ====")
            print(final_df)

        if args.summary_csv:
            summary_path = os.path.abspath(args.summary_csv)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            final_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"[ok] 汇总结果已保存至 {summary_path}")


if __name__ == "__main__":
    main()

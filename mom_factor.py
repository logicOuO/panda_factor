"""动量因子计算与评估脚本。

默认模式下，参考 base_factors.py 的清洗流程，遍历日线行情 CSV 并
写出动量相关指标（M20/M5/M1/MOM_Q）。当指定评估模式时，会聚合所有
股票的动量结果，计算与 N 日未来收益之间的日度 Rank-IC 以及整体统计。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# 与公式保持一致的稳定项，避免标准差过小导致爆炸
STD_EPS = 1e-4
EPS = 1e-12
INPUT_COLUMNS = ["datetime", "open", "close", "high", "low", "vol", "amount", "股票代码"]
FACTOR_COLUMNS = ["mom_m20", "mom_m5", "mom_m1", "mom_q"]
ICIR_ANNUAL = 1


def clip_and_fill(series: pd.Series) -> pd.Series:
    """将无穷值替换为 NaN，保留缺失以供后续流程自行处理。"""

    return series.replace([np.inf, -np.inf], np.nan)


def load_price_frame(file_path: Path) -> pd.DataFrame:
    """读取单支股票日线数据并完成基本清洗。"""

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype={"股票代码": str},
    )
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"文件 {file_path} 缺少必要列: {missing}")

    df = df.rename(columns={"股票代码": "stock_code", "股票名称": "stock_name"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    numeric_cols = ["open", "close", "high", "low", "vol", "amount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # 补齐股票代码与名称
    df["stock_code"] = df["stock_code"].str.strip().str.zfill(6)
    if "stock_name" not in df.columns:
        df["stock_name"] = df["stock_code"]
    else:
        df["stock_name"] = df["stock_name"].ffill().bfill()

    return df


def parse_date_arg(value: str) -> pd.Timestamp:
    """解析命令行日期参数，支持 YYYYMMDD / YYYY-MM-DD 等格式。"""

    try:
        return pd.to_datetime(value)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"无法解析日期 {value}: {exc}") from exc


def build_identity_frame(base_df: pd.DataFrame, include_close: bool = False) -> pd.DataFrame:
    """生成包含时间、股票标识（可选收盘价）的基础列。"""

    payload: dict[str, pd.Series] = {
        "datetime": base_df["datetime"],
        "stock_code": base_df["stock_code"],
        "stock_name": base_df.get("stock_name", base_df["stock_code"]),
    }
    if include_close:
        payload["close"] = base_df["close"]
    return pd.DataFrame(payload)


def list_input_files(input_dir: Path, limit: int | None) -> list[Path]:
    """列出输入目录内的 CSV 文件，必要时限制数量。"""

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"输入目录 {input_dir} 未找到 CSV 文件")
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def compute_mom_factors(df: pd.DataFrame) -> pd.DataFrame:
    """按照给定公式计算动量相关指标。

    公式来源：
    - M20 = C / REF(C, 20) - 1
    - M5  = C / REF(C, 5)  - 1
    - M1  = C / REF(C, 1)  - 1
    - MA20 = MA(C, 20)
    - STD20 = STD(C, 20)
    - MOM_Q = SIGN(C - MA20) * (C - MA20)^2 / (STD20^2 + 0.0001)
    """

    c = df["close"]

    m20 = c / (c.shift(20) + EPS) - 1
    m5 = c / (c.shift(5) + EPS) - 1
    m1 = c / (c.shift(1) + EPS) - 1

    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()

    price_dev = c - ma20
    mom_q = np.sign(price_dev) * (price_dev ** 2) / (std20 ** 2 + STD_EPS)

    factors = {
        "mom_m20": clip_and_fill(m20),
        "mom_m5": clip_and_fill(m5),
        "mom_m1": clip_and_fill(m1),
        "mom_q": clip_and_fill(mom_q),
    }

    return pd.DataFrame(factors, index=df.index)


def persist_factors(base_df: pd.DataFrame, factors: pd.DataFrame, output_path: Path) -> None:
    """写出动量因子结果，保留基础标识列。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export = pd.DataFrame(
        {
            "datetime": base_df["datetime"],
            "stock_code": base_df["stock_code"],
            "stock_name": base_df.get("stock_name", base_df["stock_code"]),
        }
    )
    # 降低存储空间：动量因子统一转为 float32
    export = pd.concat([export, factors.astype(np.float32).reset_index(drop=True)], axis=1)
    export.to_csv(output_path, index=False, encoding="utf-8-sig")


def process_file(file_path: Path, output_dir: Path) -> None:
    """处理单个 CSV 文件并落地。"""

    df = load_price_frame(file_path)
    factors = compute_mom_factors(df)
    output_path = output_dir / file_path.name
    persist_factors(df, factors, output_path)
    # print(f"[ok] {file_path.name} -> {output_path}")


def _process_file_mp(args: tuple[str, str]) -> None:
    file_path_str, output_dir_str = args
    process_file(Path(file_path_str), Path(output_dir_str))


def run_batch(
    input_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    processes: int | None = None,
) -> None:
    """遍历输入目录批量计算，支持多进程。"""

    files = list_input_files(input_dir, limit)

    total = len(files)
    if processes is None or processes <= 1:
        for idx, file_path in enumerate(files, start=1):
            process_file(file_path, output_dir)
            # print(f"[progress] {idx}/{total}")
        return

    ctx = mp.get_context("spawn")
    tasks = [(str(path), str(output_dir)) for path in files]
    with ctx.Pool(processes=processes) as pool:
        for idx, _ in enumerate(pool.imap_unordered(_process_file_mp, tasks), start=1):
            print(f"[progress] {idx}/{total}")


def _concat_factor_frame(df: pd.DataFrame, factors: pd.DataFrame, include_close: bool = False) -> pd.DataFrame:
    """组装基础列与因子列，评估模式可附带收盘价。"""

    base = build_identity_frame(df, include_close=include_close)
    return pd.concat([base, factors.reset_index(drop=True)], axis=1)


def add_forward_return(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """按股票分组计算未来 N 日收益率。"""

    df = df.sort_values(["stock_code", "datetime"]).reset_index(drop=True)
    future_close = df.groupby("stock_code")['close'].shift(-shift)
    df["forward_ret"] = future_close / (df["close"] + EPS) - 1
    return df


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """根据给定日期范围筛选数据。"""

    if start_date is None and end_date is None:
        return df

    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= df["datetime"] >= start_date
    if end_date is not None:
        mask &= df["datetime"] <= end_date
    return df.loc[mask].reset_index(drop=True)


def daily_ic(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """逐日计算因子与未来收益的秩相关。"""

    ic_rows: list[pd.Series] = []
    for day, sub in df.groupby("datetime"):
        correlations: dict[str, float] = {}
        for factor in factor_cols:
            pair = sub[[factor, "forward_ret"]].dropna()
            if len(pair) < 2:
                correlations[factor] = np.nan
                continue
            correlations[factor] = pair[factor].corr(pair["forward_ret"], method="spearman")

        ic_series = pd.Series(correlations, name=day)
        if ic_series.dropna().empty:
            continue
        ic_rows.append(ic_series)

    return pd.DataFrame(ic_rows)


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """汇总 IC 统计：均值/波动/ICIR/T 值/胜率。"""

    if ic_df.empty:
        return pd.DataFrame()

    mean_ic = ic_df.mean()
    std_ic = ic_df.std()
    icir = mean_ic / std_ic * np.sqrt(ICIR_ANNUAL)
    t_val = mean_ic / std_ic * np.sqrt(len(ic_df))
    win_rate = (ic_df > 0).mean()

    stats = pd.DataFrame(
        {
            "Mean_IC": mean_ic,
            "Std_IC": std_ic,
            "ICIR": icir,
            "T_val": t_val,
            "Win_Rate": win_rate,
        }
    )
    return stats.sort_values("ICIR", ascending=False)


def evaluate_mom_factors(
    input_dir: Path,
    limit: int | None,
    label_shift: int,
    report_dir: Path,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> None:
    """评估动量因子与未来 N 日收益的关系，输出日度 IC 与汇总。"""

    files = list_input_files(input_dir, limit)
    frames: list[pd.DataFrame] = []
    total = len(files)

    for idx, file_path in enumerate(files, start=1):
        df = load_price_frame(file_path)
        factors = compute_mom_factors(df)
        frame = _concat_factor_frame(df, factors, include_close=True)
        frames.append(frame)
        # print(f"[eval-load] {idx}/{total} {file_path.name}")

    all_data = pd.concat(frames, ignore_index=True)
    all_data = add_forward_return(all_data, label_shift)
    all_data = filter_by_date_range(all_data, start_date, end_date)

    if all_data.empty:
        print("[warn] 日期过滤后无可用样本，请检查 --start-date / --end-date 设置。")
        return

    ic_df = daily_ic(all_data, FACTOR_COLUMNS)
    stats = ic_summary(ic_df)

    if stats.empty:
        print("[warn] 无有效 IC 结果（可能因样本量过小或数据缺失）。")
        return

    start = all_data["datetime"].min().strftime("%Y%m%d")
    end = all_data["datetime"].max().strftime("%Y%m%d")
    prefix = f"mom_shift{label_shift}_{start}_{end}"

    report_dir.mkdir(parents=True, exist_ok=True)
    ic_path = report_dir / f"{prefix}_daily_ic.csv"
    summary_path = report_dir / f"{prefix}_ic_summary.csv"

    ic_df.to_csv(ic_path, index=True, encoding="utf-8-sig")
    stats.to_csv(summary_path, encoding="utf-8-sig")

    print("\n========= 日度 IC 汇总 =========")
    print(stats)
    print(f"\n[ok] 日度 IC 已保存到 {ic_path}")
    print(f"[ok] 汇总统计已保存到 {summary_path}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        if any("SimHei" in font.name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = ["SimHei"]
        (ic_df.cumsum() * 100).plot(figsize=(10, 6), title="Cumulative IC (%)")
        plt.tight_layout()
        fig_path = report_dir / f"{prefix}_cum_ic.png"
        plt.savefig(fig_path, dpi=200)
        print(f"[ok] 累计 IC 曲线已保存到 {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 绘制累计 IC 失败：{exc}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动量因子计算与评估")
    parser.add_argument("--mode", choices=["compute", "evaluate"], default="compute", help="compute=写出因子；evaluate=评估 IC")
    parser.add_argument("--input", default="datasets", help="原始日线数据目录")
    parser.add_argument("--output", default="output/mom_factors", help="compute 模式下的输出目录")
    parser.add_argument("--limit", type=int, default=None, help="调试用文件数上限")
    parser.add_argument("--processes", type=int, default=mp.cpu_count(), help="compute 模式下的并行进程数量")
    parser.add_argument("--label-shift", type=int, default=8, help="evaluate 模式：标签使用 N 日后收益")
    parser.add_argument("--eval-output", default="output/mom_eval", help="evaluate 模式：评估结果输出目录")
    parser.add_argument("--start-date", type=parse_date_arg, default=None, help="evaluate 模式：起始日期 (例如 20240101)")
    parser.add_argument("--end-date", type=parse_date_arg, default=None, help="evaluate 模式：结束日期 (例如 20240930)")
    return parser.parse_args(args=argv)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)

    if args.mode == "compute":
        output_dir = Path(args.output)
        run_batch(input_dir, output_dir, args.limit, args.processes)
        return

    report_dir = Path(args.eval_output)
    start_date = args.start_date
    end_date = args.end_date

    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("--start-date 不能晚于 --end-date")

    evaluate_mom_factors(input_dir, args.limit, args.label_shift, report_dir, start_date, end_date)


if __name__ == "__main__":
    main()

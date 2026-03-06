"""
自动搜索最佳 seed：在指定 seed 范围内多次训练，按 Acc 选出最佳 seed，并可选写回配置。

用法（在项目根目录 MoRE2026-Cloud 下执行）:
  python src/search_seed.py
  python src/search_seed.py --range 2020 2040
  python src/search_seed.py --seeds 42,123,2024,2025
  python src/search_seed.py --range 2020 2030 --metric acc --write-yaml
"""
import argparse
import gc
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 确保从项目根目录运行时可导入 src 下模块
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from loguru import logger
from hydra import compose, initialize_config_dir

from main import Trainer, MoRERADARTrainer
from utils.core_utils import set_seed

# 每 N 个 seed 保留该段内最优模型并释放显存/内存
_RELEASE_EVERY_N = 100

# 追加表格时的指标列顺序（与 _render_metrics_table 一致）
_METRIC_COLS = ["acc", "macro_f1", "macro_prec", "macro_rec", "a_f1", "b_f1", "c_f1"]


def run_one_seed(cfg, seed: int, save_path: Path):
    """用指定 seed 跑一轮训练，返回 metrics 字典。"""
    set_seed(seed)
    if cfg.para.get("use_radar", False):
        trainer = MoRERADARTrainer(cfg)
    else:
        trainer = Trainer(cfg)
    trainer.save_path = save_path
    save_path.mkdir(parents=True, exist_ok=True)
    return trainer.run()

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _metrics_row(seed: int, metrics: dict) -> tuple[str, str]:
    """返回 (md_row, csv_row)，用于追加到表格。列顺序：seed, acc, macro_f1, ..."""
    md_vals = [str(seed)] + [
        "" if _to_float(metrics.get(k)) is None else f"{_to_float(metrics.get(k)):.5f}"
        for k in _METRIC_COLS
    ]
    csv_vals = [str(seed)] + [
        "" if _to_float(metrics.get(k)) is None else f"{_to_float(metrics.get(k)):.8f}"
        for k in _METRIC_COLS
    ]
    md_row = "| " + " | ".join(md_vals) + " |"
    csv_row = ",".join(csv_vals)
    return md_row, csv_row


def _append_result_to_files(seed: int, metrics: dict, summary_md: Path, summary_csv: Path, first: bool):
    """每训练一个 seed 后追加一行。first=True 时写表头+分隔行+首行；否则仅追加一行。"""
    md_header = "| seed | " + " | ".join(_METRIC_COLS) + " |"
    md_sep = "| " + " | ".join(["---"] * (1 + len(_METRIC_COLS))) + " |"
    csv_header = "seed," + ",".join(_METRIC_COLS)
    md_row, csv_row = _metrics_row(seed, metrics)
    if first:
        summary_md.write_text(md_header + "\n" + md_sep + "\n" + md_row + "\n", encoding="utf-8")
        summary_csv.write_text(csv_header + "\n" + csv_row + "\n", encoding="utf-8")
    else:
        with summary_md.open("a", encoding="utf-8") as f:
            f.write(md_row + "\n")
        with summary_csv.open("a", encoding="utf-8") as f:
            f.write(csv_row + "\n")


def _render_metrics_table(results: list[tuple[int, dict]], metric_order: list[str] | None = None):
    """
    返回 (markdown_table_str, csv_str)。
    表格格式：列=seed；行=各项指标（acc/macro_f1/...）。
    """
    seeds = [s for s, _ in results]
    all_keys = set()
    for _, m in results:
        if isinstance(m, dict):
            all_keys.update(m.keys())

    preferred = metric_order or [
        "acc",
        "macro_f1",
        "macro_prec",
        "macro_rec",
        "a_f1",
        "b_f1",
        "c_f1",
    ]
    rows = [k for k in preferred if k in all_keys] + sorted([k for k in all_keys if k not in preferred])

    # Markdown
    header = ["metric"] + [str(s) for s in seeds]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for k in rows:
        line = [k]
        for _, m in results:
            v = _to_float(m.get(k)) if isinstance(m, dict) else None
            line.append("" if v is None else f"{v:.5f}")
        md_lines.append("| " + " | ".join(line) + " |")
    md_table = "\n".join(md_lines)

    # CSV（同样行=metric，列=seed）
    csv_lines = [",".join(header)]
    for k in rows:
        line = [k]
        for _, m in results:
            v = _to_float(m.get(k)) if isinstance(m, dict) else None
            line.append("" if v is None else f"{v:.8f}")
        csv_lines.append(",".join(line))
    csv_text = "\n".join(csv_lines) + "\n"
    return md_table, csv_text


def search_seeds(
    config_name: str = "HateMM_MoRE",
    seeds: list = None,
    metric: str = "acc",
    write_yaml: bool = False,
    log_level: str = "WARNING",
):
    """
    config_name: Hydra 配置名（不含 .yaml）
    seeds: 要尝试的 seed 列表
    metric: 用于选最优的指标，默认 'acc'，可选 'macro_f1'
    write_yaml: 是否将最佳 seed 写回 HateMM_MoRE.yaml
    log_level: 搜索时主进程日志级别，减少刷屏
    """
    if seeds is None:
        seeds = list(range(2020, 2040))
    logger.remove()
    logger.add(sys.stdout, level=log_level)

    config_dir = str(_project_root / "src" / "config")
    log_base = _project_root / "src" / "log"
    log_base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_md = log_base / f"seed_search_summary_{ts}.md"
    summary_csv = log_base / f"seed_search_summary_{ts}.csv"

    results = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for i, seed in enumerate(seeds):
            logger.info(f"Seed search [{i+1}/{len(seeds)}] seed={seed}")
            cfg = compose(config_name=config_name, overrides=[f"seed={seed}"])
            save_path = log_base / f"seed_search_{seed}"
            try:
                metrics = run_one_seed(cfg, seed, save_path)
                results.append((seed, metrics))
                logger.info(
                    f"  seed={seed} -> {metric}={metrics.get(metric, 0):.4f} acc={metrics.get('acc', 0):.4f}"
                )
            except Exception as e:
                logger.warning(f"  seed={seed} failed: {e}")
                results.append((seed, {metric: -1.0, "acc": -1.0}))

            # 每训练一个 seed 立刻追加一行到 MD/CSV（追加模式）
            _append_result_to_files(
                seed, results[-1][1], summary_md, summary_csv, first=(len(results) == 1)
            )

            # 每 100 个 seed：只保留该段内最优模型，删除其余 99 个目录，并释放显存/内存
            if (i + 1) % _RELEASE_EVERY_N == 0 and len(results) >= _RELEASE_EVERY_N:
                block_results = results[-(_RELEASE_EVERY_N):]
                valid_block = [(s, m) for s, m in block_results if m.get(metric, -1) >= 0]
                if valid_block:
                    best_in_block = max(valid_block, key=lambda x: x[1].get(metric, -1))[0]
                    for s, _ in block_results:
                        if s != best_in_block:
                            d = log_base / f"seed_search_{s}"
                            if d.exists():
                                try:
                                    shutil.rmtree(d)
                                    logger.info(f"  Removed non-best dir: {d}")
                                except Exception as e:
                                    logger.warning(f"  Failed to remove {d}: {e}")
                    logger.info(f"  Kept best model in block: seed={best_in_block}")
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
                logger.info(f"  [Released GPU/memory after {i+1} runs]")

    if not results:
        logger.error("No successful runs.")
        return None

    # 按选定指标选最优
    valid = [(s, m) for s, m in results if m.get(metric, -1) >= 0]
    if not valid:
        logger.error("No valid metrics.")
        return None
    best_seed, best_metrics = max(valid, key=lambda x: x[1].get(metric, -1))
    best_val = best_metrics.get(metric, 0)

    logger.info("")
    logger.info(f"Best seed by {metric}: {best_seed} ({metric}={best_val:.4f}, acc={best_metrics.get('acc', 0):.4f})")
    logger.info("All results (seed, {}): {}", metric, [(s, m.get(metric)) for s, m in results])
    logger.info("Seed metrics table (appended per seed): {} / {}", summary_md, summary_csv)
    md_table, _ = _render_metrics_table(results)
    logger.info("\n" + md_table)

    if write_yaml:
        yaml_path = _project_root / "src" / "config" / "HateMM_MoRE.yaml"
        if not yaml_path.exists():
            logger.warning(f"Config not found: {yaml_path}, skip writing.")
        else:
            text = yaml_path.read_text(encoding="utf-8")
            new_text = re.sub(r"^seed:\s*\d+\s*$", f"seed: {best_seed}", text, flags=re.MULTILINE)
            if new_text != text:
                yaml_path.write_text(new_text, encoding="utf-8")
                logger.info(f"Updated {yaml_path} with seed: {best_seed}")
            else:
                logger.warning("Could not find 'seed: <number>' line in yaml to replace.")

    return best_seed


def main():
    import os
    os.chdir(_project_root)  # 保证 config / 数据等路径正确
    parser = argparse.ArgumentParser(description="Search best seed for HateMM_MoRE")
    parser.add_argument("--config", default="HateMM_MoRE", help="Config name without .yaml")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="Seed range [start, end)")
    parser.add_argument("--seeds", type=str, help="Comma-separated seeds, e.g. 42,123,2024")
    parser.add_argument("--metric", default="acc", choices=["macro_f1", "acc"], help="Metric to maximize")
    parser.add_argument("--write-yaml", action="store_true", help="Write best seed to HateMM_MoRE.yaml")
    parser.add_argument("--log-level", default="WARNING", help="Log level during search (e.g. INFO, WARNING)")
    args = parser.parse_args()

    if args.range:
        seeds = list(range(args.range[0], args.range[1]))
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = list(range(2020, 2040))

    search_seeds(
        config_name=args.config,
        seeds=seeds,
        metric=args.metric,
        write_yaml=args.write_yaml,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

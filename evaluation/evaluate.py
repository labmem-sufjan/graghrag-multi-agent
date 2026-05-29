"""对已保存的 jsonl 跑分：规则指标（must_contain、gold_chunk_recall）+ 可选 Ragas。

不会重新调用 LLM 答题，只读 evaluation/outputs 里的历史运行结果。
  python -m evaluation.evaluate --runs naive_rag,multi_agent --no-ragas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import OUTPUTS_DIR
from evaluation.dataset import load_dataset
from evaluation.io import latest_run_path, load_run_jsonl
from evaluation.metrics.rule_metrics import aggregate_rule_metrics, score_run_result
from evaluation.metrics.ragas_metrics import run_ragas_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _must_contain_map(samples) -> dict[str, list[str]]:
    return {s.id: s.must_contain for s in samples}


def _gold_chunk_map(samples) -> dict[str, list[str]]:
    return {s.id: s.gold_chunk_ids for s in samples if s.gold_chunk_ids}


def _record_to_result(record: dict) -> "RunResult":
    from evaluation.schemas import RunResult

    kwargs = {f: record.get(f) for f in RunResult.__dataclass_fields__}
    return RunResult(**kwargs)


def evaluate_file(
    path: Path,
    must_map: dict[str, list[str]],
    gold_map: dict[str, list[str]],
    use_ragas: bool,
) -> dict:
    records = load_run_jsonl(path)
    rule_rows = [
        score_run_result(
            _record_to_result(r),
            must_map.get(r["id"], []),
            gold_map.get(r["id"], []),
        )
        for r in records
    ]

    report = {
        "file": str(path),
        "pipeline": records[0].get("pipeline") if records else "",
        "rule_summary": aggregate_rule_metrics(rule_rows),
        "rule_per_item": rule_rows,
    }
    if use_ragas:
        report["ragas"] = run_ragas_evaluation(records)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved run JSONL files")
    parser.add_argument(
        "--runs",
        type=str,
        default="naive_rag,multi_agent",
        help="Pipelines — use latest run_* file for each",
    )
    parser.add_argument("--file", type=Path, action="append", help="Explicit JSONL path(s)")
    parser.add_argument("--no-ragas", action="store_true", help="Skip Ragas (faster)")
    parser.add_argument("--out", type=Path, default=None, help="Report JSON path")
    args = parser.parse_args()

    samples = load_dataset()
    must_map = _must_contain_map(samples)
    gold_map = _gold_chunk_map(samples)

    paths: list[Path] = list(args.file or [])
    if not paths:
        for p in [x.strip() for x in args.runs.split(",")]:
            latest = latest_run_path(p)
            if latest:
                paths.append(latest)
            else:
                logger.warning("No run file for pipeline %s", p)

    reports = [
        evaluate_file(path, must_map, gold_map, use_ragas=not args.no_ragas)
        for path in paths
    ]

    out_path = args.out or (OUTPUTS_DIR / "report_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report written to %s", out_path)

    for rep in reports:
        print(f"\n=== {rep.get('pipeline')} ({rep.get('file')}) ===")
        for k, v in rep.get("rule_summary", {}).items():
            print(f"  {k}: {v}")
        ragas = rep.get("ragas", {})
        if ragas.get("summary"):
            print("  Ragas:")
            for k, v in ragas["summary"].items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


if __name__ == "__main__":
    main()

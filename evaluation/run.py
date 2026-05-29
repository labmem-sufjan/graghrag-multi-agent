"""批量跑评测：对 test_dataset.json 每题调用 pipeline，结果写入 outputs/*.jsonl。

在项目根目录执行：
  python -m evaluation.run --pipelines naive_rag,multi_agent
  python -m evaluation.run --pipeline naive_rag --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import ALL_PIPELINES, PIPELINE_AGENT, PIPELINE_NAIVE
from evaluation.dataset import load_dataset
from evaluation.io import save_run_jsonl
from evaluation.pipelines import run_multi_agent, run_naive_rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RUNNERS = {
    PIPELINE_NAIVE: run_naive_rag,
    PIPELINE_AGENT: run_multi_agent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipelines on test set")
    parser.add_argument(
        "--pipelines",
        type=str,
        default="naive_rag,multi_agent",
        help=f"Comma-separated: {', '.join(ALL_PIPELINES)}",
    )
    parser.add_argument("--pipeline", type=str, help="Single pipeline (alias for --pipelines)")
    parser.add_argument("--limit", type=int, default=None, help="Only first N questions")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to test_dataset.json")
    parser.add_argument("--tag", type=str, default=None, help="Output file tag")
    args = parser.parse_args()

    pipe_str = args.pipeline or args.pipelines
    pipelines = [p.strip() for p in pipe_str.split(",") if p.strip()]
    for p in pipelines:
        if p not in _RUNNERS:
            raise SystemExit(f"Unknown pipeline: {p}. Choose from {ALL_PIPELINES}")

    samples = load_dataset(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    logger.info("Loaded %s samples", len(samples))

    for pipeline in pipelines:
        runner = _RUNNERS[pipeline]
        results = []
        for i, sample in enumerate(samples, 1):
            logger.info("[%s] %s/%s %s", pipeline, i, len(samples), sample.id)
            results.append(runner(sample))
        out = save_run_jsonl(results, pipeline, tag=args.tag)
        errors = sum(1 for r in results if r.error)
        logger.info("Saved %s (%s errors)", out, errors)


if __name__ == "__main__":
    main()

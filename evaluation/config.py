"""Evaluation-specific paths and defaults."""

from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = EVAL_ROOT / "outputs"
DATASET_PATH = EVAL_ROOT / "test_dataset.json"

PIPELINE_NAIVE = "naive_rag"
PIPELINE_AGENT = "multi_agent"
ALL_PIPELINES = (PIPELINE_NAIVE, PIPELINE_AGENT)

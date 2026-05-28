# 评测模块

与在线问答 `src/` **分离**，仅负责批量跑题、打分、出报告。

## 目录结构

```text
evaluation/
├── README.md              # 本说明
├── test_dataset.json      # 测试集（问题 + ground_truth + must_contain）
├── config.py              # 路径与 pipeline 名称
├── schemas.py             # EvalSample / RunResult 数据结构
├── dataset.py             # 加载测试集
├── io.py                  # 读写 outputs/*.jsonl
├── run.py                 # 批量跑 pipeline
├── evaluate.py            # 对 jsonl 打分（规则 + Ragas）
├── pipelines/
│   ├── naive_rag.py       # 基线 B0：仅 Chroma + LLM
│   └── multi_agent.py     # 完整 Multi-Agent 系统
├── metrics/
│   ├── rule_metrics.py    # must_contain、中文率、延迟
│   └── ragas_metrics.py   # Faithfulness / Relevancy / Correctness
└── outputs/               # 运行结果（已 gitignore）
```

## 使用流程

```bash
# 1. 确保已建库，Ollama + Neo4j 已启动，.env 中 OLLAMA_LLM_MODEL=qwen2.5:7b

# 2. 批量生成答案（可先 --limit 3 试跑）
python -m evaluation.run --pipelines naive_rag,multi_agent --limit 5

# 3. 全量
python -m evaluation.run --pipelines naive_rag,multi_agent

# 4. 评分（规则指标 + Ragas，Ragas 较慢）
python -m evaluation.evaluate --runs naive_rag,multi_agent

# 仅规则指标，跳过 Ragas
python -m evaluation.evaluate --runs naive_rag,multi_agent --no-ragas
```

报告默认写入 `evaluation/outputs/report_latest.json`。

## 对比说明

| Pipeline | 含义 |
|----------|------|
| `naive_rag` | Chroma Top-K + 同一套回答 Prompt（无 Router / 图 / Critic） |
| `multi_agent` | `src.graph_workflow.run_query` 全链路 |

对比时保持 **同一 LLM、同一 embedding、同一测试集**。

## 指标

| 类型 | 指标 |
|------|------|
| 规则 | `must_contain_rate`、`gold_chunk_recall`（有标注 chunk 的题）、中文率、延迟、错误率 |
| Ragas | `faithfulness`、`answer_relevancy`、`answer_correctness` |

`gold_chunk_recall`：检索返回的 `chunk_ids` 与 `test_dataset.json` 中 `gold_chunk_ids` 的交集比例，用于衡量是否命中 p57/p58 等关键段落。

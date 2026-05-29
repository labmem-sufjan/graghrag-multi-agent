# 评测模块

与在线问答 `src/` **分离**：批量跑题、规则打分、可选 Ragas，用于对比 **Naive RAG** 与 **Multi-Agent** 全链路，并记录每次代码优化后的指标变化。

- 测试文档：`data/raw_docs/yushu.pdf`
- 测试集：`test_dataset.json`（13 道 QA，含 `must_contain`、`gold_chunk_ids`）
- 结果目录：`outputs/`（jsonl + `report_latest.json`，已 gitignore）

---

## 目录结构

```text
evaluation/
├── README.md                 # 本说明（指标与迭代）
├── test_dataset.json         # 问题、标准答案、必含关键词、金标 chunk
├── config.py                 # 路径与 pipeline 名称
├── schemas.py                # EvalSample / RunResult
├── dataset.py                # 加载测试集
├── io.py                     # 读写 outputs/*.jsonl
├── run.py                    # 批量跑 pipeline
├── evaluate.py               # 对 jsonl 打分
├── pipelines/
│   ├── naive_rag.py          # 基线 B0：仅 Chroma + LLM
│   └── multi_agent.py        # 完整 Multi-Agent（graph_workflow）
├── metrics/
│   ├── rule_metrics.py       # must_contain、gold_chunk_recall、延迟等
│   └── ragas_metrics.py      # Faithfulness / Relevancy / Correctness
└── outputs/                  # 运行产物
```

---

## 使用流程

```bash
# 前置：已建库，Neo4j + Ollama 已启动，.env 中 OLLAMA_LLM_MODEL=qwen2.5:7b

# 1. 试跑
python -m evaluation.run --pipelines naive_rag,multi_agent --limit 3

# 2. 全量生成答案
python -m evaluation.run --pipelines naive_rag,multi_agent

# 3. 打分（规则指标，较快）
python -m evaluation.evaluate --runs naive_rag,multi_agent --no-ragas

# 4. 含 Ragas（较慢，需安装 ragas 相关依赖）
python -m evaluation.evaluate --runs naive_rag,multi_agent
```

报告默认写入 `outputs/report_latest.json`。指定某次运行：

```bash
python -m evaluation.evaluate --file evaluation/outputs/run_multi_agent_20260528_160440.jsonl --no-ragas
```

---

## Pipeline 对比说明

| Pipeline | 实现 | 用途 |
|----------|------|------|
| `naive_rag` | Chroma Top-K + 统一回答 Prompt | 基线 B0，无 Router / 图 / Critic |
| `multi_agent` | `src.graph_workflow.run_query` | 完整系统 |

对比时保持 **同一 LLM、同一 embedding、同一测试集**。

---

## 指标说明

### 规则指标（`metrics/rule_metrics.py`）

| 指标 | 含义 |
|------|------|
| `must_contain_rate` | 答案是否包含 `test_dataset.json` 中规定的必含关键词（如「王兴兴」「60.13%」） |
| `gold_chunk_recall` | 检索返回的 `chunk_ids` 与 `gold_chunk_ids` 的交集比例（仅标注了 gold 的题参与汇总） |
| `chinese_rate` | 回答是否为中文（排除常见英文模板短语） |
| `latency_ms_avg` | 单题端到端耗时 |
| `error_rate` | 运行异常比例 |

`gold_chunk_recall` 用于衡量是否命中 **p57/p58** 等关键段落，与「答对关键词」互补：检索对了但生成漏写，must_contain 仍会偏低。

### Ragas（可选）

| 指标 | 含义 |
|------|------|
| `faithfulness` | 答案是否忠于检索上下文 |
| `answer_relevancy` | 答案与问题的相关度 |
| `answer_correctness` | 与标准答案的语义接近度 |

---

## 指标迭代记录（13 题全量，本地 2026-05-28）

本项目按「改代码 → 全量跑 `evaluation.run` → `evaluation.evaluate`」闭环优化。下表为仓库内保留的两次 Multi-Agent 全量结果与 Naive 基线对比。

### 全量汇总

| 阶段 | 运行文件（示例） | must_contain ↑ | gold_chunk_recall ↑ | 平均延迟 |
|------|------------------|----------------|---------------------|----------|
| 基线 | `run_naive_rag_*_144816.jsonl` | **26.9%** | **0%**（4 题有 gold） | **8.6s** |
| Multi-Agent v1 | `run_multi_agent_*_145355.jsonl` | 35.9% | 62.5% | 26.1s |
| **Multi-Agent v2** | `run_multi_agent_*_160440.jsonl` | **43.6%** | **62.5%** | **22.4s** |

- 相对 Naive，v2 **must_contain +16.7 个百分点**
- 相对 v1，v2 **must_contain +7.7 个百分点**，延迟 **-3.7s**

### 演进路线（与代码改动的对应关系）

```text
B0 naive_rag
  └─ Chroma Top-K + LLM

Multi-Agent v1
  └─ + LangGraph Router / Graph / Hybrid
  └─ + Qwen2.5 中文生成、规则 Critic
  └─ + 子公司/控制人定向 chunk 检索

Multi-Agent v2（当前推荐对比版本）
  └─ + gold_chunk_recall 评测指标
  └─ + 风险题 → hybrid + search_risk_chunks
  └─ + context_max_chunks / 单段字数上限（降噪声、降延迟）
```

### 关键优化项与代表效果

| 迭代项 | 代码位置（概要） | 代表效果 |
|--------|------------------|----------|
| 中文模型与 Prompt | `config/settings.py`、`generator` | `chinese_rate` 100% |
| 多路检索 | `graph_expert`、`hybrid`、`neo4j_client.search_*_chunks` | q002 不再误答保荐人；q011/q012 gold **100%** |
| 评测体系 | `evaluation/metrics/rule_metrics.py` | 可量化 p57/p58 命中 |
| 风险题路由 | `router.py`、`search_risk_chunks` | **q017：0% → 100%** must_contain |
| 上下文截断 | `context_limit.py`、`CONTEXT_MAX_CHUNKS` | 延迟 26.1s → **22.4s** |
| 规则 Critic | `critic.py`、`CRITIC_USE_LLM=false` | 稳定校验，无 LLM Critic 异常 |

### 分题表现（Multi-Agent v2）

| 题号 | 主题 | must_contain | gold_chunk | 备注 |
|------|------|--------------|------------|------|
| q001 | 主营业务 | 100% | — | |
| q002 | 法人/控制人 | 50% | 50% | 有王兴兴，缺 23.8216% |
| q003 | 成立/地址 | 0% | — | 待发行人基本情况定向检索 |
| q004 | 募投项目 | 50% | — | |
| q006 | 2025 营收/扣非 | 0% | — | 待财务表定向检索 |
| q007 | 毛利率 | 0% | — | |
| q009 | 研发人数 | 0% | — | |
| q011 | 8 家子公司 | 67% | **100%** | 列了 5 家，答案未写「8」 |
| q012 | 控制人+子公司 | **100%** | **100%** | |
| q015 | 人形出货量 | 0% | — | |
| q017 | 风险因素 | **100%** | — | v1→v2 提升最明显 |
| q020 | Q1 扣非同比 | **100%** | — | |
| q024 | 子公司业务 | 0% | 0% | 未命中 p58 业务描述 |

### Ragas（参考，同批 Naive 全量）

| Pipeline | faithfulness | answer_relevancy |
|----------|--------------|------------------|
| naive_rag | 0.41 | 0.42 |
| multi_agent（v1 批次） | 0.33 | **0.53** |

Ragas 与规则指标可能不一致：例如检索命中但生成漏写时，faithfulness 尚可而 must_contain 偏低。以 **`must_contain` + `gold_chunk_recall`** 为主跟踪迭代，Ragas 作辅助。

---

## 如何更新本页数字

每次改检索/路由/生成逻辑后：

```bash
python -m evaluation.run --pipelines multi_agent
python -m evaluation.evaluate --runs naive_rag,multi_agent --no-ragas
```

将 `report_latest.json` 中 `rule_summary` 同步到上文表格，或在 PR 中附上 `outputs/run_*.jsonl` 文件名便于追溯。

---

## 测试集字段说明

`test_dataset.json` 中每道题包含：

| 字段 | 说明 |
|------|------|
| `question` | 用户问题 |
| `ground_truth` | 参考答案（Ragas / 人工对照） |
| `must_contain` | 答案应包含的关键词列表 |
| `gold_chunk_ids` | 期望检索命中的 chunk_id（可选） |
| `route_hint` | 期望路由（vector/graph/hybrid），用于设计而非自动打分 |
| `category` | factual / governance / financial / risk 等 |

扩充测试集时建议同时补充 `must_contain` 与关键页的 `gold_chunk_ids`，便于区分「检索失败」与「生成失败」。

---

## 待优化（评测侧跟踪）

- [x] 发行人配置 `config/document_profile.yml`（Critic / 检索去硬编码）
- [x] 财务、发行人基本情况定向 chunk（`search_financial_chunks` / `search_issuer_profile_chunks`）
- [ ] 重跑全量评测，更新上文指标表（v3）
- [ ] q024 与 p58 子公司业务表对齐
- [ ] q002 生成约束带出持股比例
- [ ] 全量 Ragas 复评

换文档评测时：复制 `document_profile.example.yml`，改实体名后设置 `DOCUMENT_PROFILE_PATH`，并更新 `test_dataset.json` 的 `must_contain` / `gold_chunk_ids`。

# 代码阅读地图

按数据流顺序阅读，便于一次性弄懂全仓库。

## 1. 离线建库

| 文件 | 作用 |
|------|------|
| `src/pipeline/document_processor.py` | PDF → 分块，生成 `chunk_id` |
| `src/pipeline/extractor.py` | 每 chunk LLM 抽实体/关系 → Neo4j |
| `src/pipeline/build_knowledge.py` | CLI：Chroma 索引 + 图谱抽取 |
| `src/tools/vector_client.py` | Chroma 写入与相似检索 |
| `src/tools/neo4j_client.py` | 图谱 schema、抽取落库、在线 chunk 搜索 |

## 2. 配置

| 文件 | 作用 |
|------|------|
| `config/settings.py` | 环境变量、Top-K、上下文上限 |
| `config/document_profile.yml` | 发行人、子公司、检索扩展（换 PDF 改这里） |
| `config/prompts.py` | 抽取 / 路由 / 回答 / Critic 的系统 Prompt |

## 3. 在线问答

| 文件 | 作用 |
|------|------|
| `src/graph_workflow.py` | LangGraph 编排 |
| `src/agents/router.py` | 选 vector / graph / hybrid |
| `src/agents/vector_expert.py` | 纯向量 |
| `src/agents/graph_expert.py` | 纯图谱 + 定向 chunk |
| `src/agents/hybrid.py` | 向量 + 图谱 |
| `src/agents/generator.py` | 生成答案 |
| `src/agents/critic.py` | 规则质检 |
| `src/tools/retrieval.py` | 问题 → 检索关键词 |
| `src/tools/directed_retrieval.py` | 问题类型 → 定向 chunk |
| `src/tools/context_limit.py` | 截断送入 LLM 的上下文 |

## 4. 评测

| 文件 | 作用 |
|------|------|
| `evaluation/test_dataset.json` | 13 道测试题 |
| `evaluation/run.py` | 批量跑 naive / multi_agent |
| `evaluation/evaluate.py` | 对 jsonl 打分 |
| `evaluation/metrics/rule_metrics.py` | must_contain、gold_chunk_recall |

## 5. 入口

| 文件 | 作用 |
|------|------|
| `app.py` | CLI |
| `web_app.py` | Streamlit |

更细的指标与迭代记录见 [evaluation/README.md](../evaluation/README.md)。

# GraphRAG Multi-Agent

基于 **LangGraph** 的招股书 GraphRAG 多智能体问答系统：Chroma 向量检索 + Neo4j 知识图谱 + Ollama 本地大模型，支持 Router 分诊、多路检索、规则 Critic 与可复现评测。

> 测试文档：`data/raw_docs/yushu.pdf`（宇树科技招股书）  
> 评测集：13 道结构化 QA（`evaluation/test_dataset.json`）

---

## 架构

```text
用户问题 → Router（规则 + LLM）→ Vector | Graph | Hybrid Expert → Generate → Critic → 回答
                ↑                        ↑
         风险/子公司/控制人           关键词定向 chunk + 图谱子图
```

| 组件 | 职责 |
|------|------|
| **Router** | 按问题类型路由：治理/子公司 → graph/hybrid；风险列举 → hybrid；其余 → vector |
| **Vector Expert** | Chroma Top-K 语义检索 |
| **Graph Expert** | Neo4j 关键词 chunk + 实体子图 + 子公司关系 |
| **Hybrid Expert** | 向量 + 定向 chunk（子公司/控制人/风险）+ 图谱扩展 |
| **Critic** | 规则校验（中文、禁幻觉人名等），默认不额外调 LLM |

---

## 迭代与指标提升

本项目按「**可评测 → 可对比 → 可定位 → 可优化**」迭代，每次改动都在同一测试集上复跑并记录结果。

### 演进路线

```text
B0 Naive RAG          →  Multi-Agent v1          →  Multi-Agent v2（当前）
Chroma + LLM              + Router/Graph/Hybrid       + 定向检索 / 上下文截断
                          + Qwen2.5 中文              + gold_chunk 指标
                          + 规则 Critic               + 风险题 hybrid 路由
```

### 全量指标对比（13 题，2026-05-28 本地评测）

| 阶段 | Pipeline | must_contain ↑ | gold_chunk_recall ↑ | 平均延迟 |
|------|----------|----------------|---------------------|----------|
| 基线 | `naive_rag` | 26.9% | 0%（4 题有标注） | **8.6s** |
| v1 | `multi_agent`（首版全链路） | 35.9% | 62.5% | 26.1s |
| **v2** | **`multi_agent`（优化后）** | **43.6%** | **62.5%** | **22.4s** |

相对基线，v2 的 **must_contain 提升 +16.7 个百分点**；相对 v1，**+7.7 个百分点**，同时延迟下降 **3.7s**（上下文上限与图谱 hop 收紧）。

### 关键优化项与效果

| 迭代项 | 做法 | 代表效果 |
|--------|------|----------|
| 模型与生成 | `OLLAMA_LLM_MODEL=qwen2.5:7b`，生成 Prompt 强制简体中文 | 中文率 100%，告别英文模板回答 |
| 多路检索 | 子公司/控制人定向 `search_*_chunks`，hybrid 合并向量+图谱 | q002 不再误答保荐人；q011/q012 gold recall **100%** |
| 评测体系 | `evaluation/` 独立模块：naive 基线、`must_contain`、`gold_chunk_recall` | 可量化「是否命中 p57/p58」 |
| 风险题路由 | 风险关键词 → **hybrid** + `search_risk_chunks` | **q017：0% → 100%**（能列举四项风险） |
| 上下文治理 | `context_max_chunks=10`、单段 800 字上限、关键词片段优先 | 延迟 26.1s → **22.4s**，减少书末噪声 chunk |
| Critic | 默认规则 Critic（`CRITIC_USE_LLM=false`） | 稳定通过校验，避免 LLM Critic 异常 |

### 分题型表现（v2 vs naive）

| 题型 | 典型题号 | v2 表现 | 说明 |
|------|----------|---------|------|
| 治理/股权 | q002, q012 | must **50%–100%**，gold **50%–100%** | 控制人、子公司表检索稳定 |
| 结构/子公司 | q011 | must 67%，gold **100%** | 能列 5 家子公司；「8」字未进答案 |
| 风险 | q017 | must **100%** | 优化前后差异最大的一题 |
| 财务/经营 | q006, q007, q009, q015 | 仍偏弱 | 依赖向量命中表格页，待加财务定向检索 |
| 基线对比 | 全集 | naive 26.9% vs v2 **43.6%** | Multi-Agent 在图谱类问题上优势最明显 |

复现最新数字：

```bash
python -m evaluation.run --pipelines naive_rag,multi_agent
python -m evaluation.evaluate --runs naive_rag,multi_agent --no-ragas
# 报告：evaluation/outputs/report_latest.json
```

---

## 快速开始

### 前置条件

- Docker：`docker compose up -d`（Neo4j）
- Ollama：`ollama pull qwen2.5:7b` 与 `ollama pull nomic-embed-text`
- 复制 `.env.example` → `.env`，确认 `OLLAMA_LLM_MODEL=qwen2.5:7b`

### 建库（首次）

```bash
pip install -r requirements.txt
python -m src.pipeline.build_knowledge --source data/raw_docs/yushu.pdf
```

### 问答

```bash
# Web（推荐）
pip install streamlit
streamlit run web_app.py

# CLI
python app.py "公司的实际控制人是谁？有哪些全资子公司？"
```

### 示例问题

| 路由 | 示例 |
|------|------|
| vector | 2025 年公司营业收入和扣非净利润分别是多少？ |
| graph | 公司有多少家全资子公司？实际控制人是谁？ |
| hybrid | 招股书列举了哪些与发行人相关的风险？至少四项。 |

---

## 项目结构

```text
src/agents/          router, vector/graph/hybrid expert, generator, critic
src/graph_workflow.py
src/tools/           Chroma、Neo4j、retrieval、context_limit
src/pipeline/        PDF 分块、图谱抽取、建库 CLI
evaluation/          测试集、naive 基线、规则指标 + Ragas
app.py / web_app.py  CLI / Streamlit
config/              settings、prompts
```

架构细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。评测说明见 [evaluation/README.md](evaluation/README.md)。

---

## 配置

`.env.example` 中常用项：

| 变量 | 说明 |
|------|------|
| `OLLAMA_LLM_MODEL` | 建议 `qwen2.5:7b` |
| `RETRIEVAL_TOP_K` | 向量 Top-K，默认 5 |
| `CONTEXT_MAX_CHUNKS` | 送入 LLM 的最大 chunk 数，默认 10 |
| `GRAPH_HOP_CHUNKS` | 图谱扩展 chunk 上限，默认 6 |
| `CRITIC_USE_LLM` | 默认 `false`，仅用规则 Critic |

---

## 后续方向

- [ ] 财务/发行人基本情况定向 chunk（q003、q006、q007）
- [ ] q024 子公司业务描述与 p58 表对齐
- [ ] 生成阶段强化 must 数字（如持股比例 23.8216%）
- [ ] Ragas 全量复评，观察 Faithfulness 随上下文截断的变化

---

## License

MIT（如适用请按仓库实际许可证填写）

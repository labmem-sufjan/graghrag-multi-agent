# GraphRAG Multi-Agent

企业招股书场景的 **GraphRAG + 多智能体** 问答系统：离线将 PDF 写入 Chroma 向量库与 Neo4j 知识图谱，在线由 LangGraph 编排 Router、多路检索专家、生成与 Critic。

当前样例文档：`data/raw_docs/yushu.pdf`（宇树科技招股书）。

---

## 特性

- **双存储检索**：Chroma 语义检索 + Neo4j 实体/关系与子图扩展
- **多智能体分诊**：按问题类型路由 Vector / Graph / Hybrid 专家
- **定向检索**：子公司表、控制人、风险因素等关键词优先 chunk（见 `src/tools/neo4j_client.py`）
- **规则 Critic**：默认不额外调用 LLM，校验中文与明显幻觉
- **独立评测模块**：基线对比、规则指标与 Ragas，见 [evaluation/README.md](evaluation/README.md)

---

## 架构概览

```text
[离线] PDF → 分块 → 向量入库(Chroma) + 图谱抽取(Neo4j)

[在线] 用户问题 → Router → Vector | Graph | Hybrid Expert → Generate → Critic → 回答
```

| 模块 | 说明 |
|------|------|
| Router | 规则关键词 + LLM，选择检索策略 |
| Vector Expert | Chroma Top-K |
| Graph Expert | 关键词 chunk、实体子图、子公司关系 |
| Hybrid Expert | 向量 + 定向 chunk + 图谱扩展 |
| Critic | 规则校验（可配置 `CRITIC_USE_LLM`） |

更细的目录与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 快速开始

### 1. 环境

```bash
cp .env.example .env
docker compose up -d          # Neo4j
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
pip install -r requirements.txt
```

`.env` 中建议设置 `OLLAMA_LLM_MODEL=qwen2.5:7b`（中文生成效果明显优于 llama3）。

### 2. 建库（首次）

```bash
python -m src.pipeline.build_knowledge --source data/raw_docs/yushu.pdf
```

支持断点续跑：`--resume`、`--skip-chroma`、`--skip-graph`。

### 3. 问答

```bash
# Web
pip install streamlit
streamlit run web_app.py

# CLI
python app.py "公司的实际控制人是谁？"
python app.py   # 交互模式
```

### 4. 评测（可选）

批量跑题与指标报告在 **`evaluation/`** 目录，不与在线代码耦合：

```bash
python -m evaluation.run --pipelines naive_rag,multi_agent
python -m evaluation.evaluate --runs naive_rag,multi_agent --no-ragas
```

指标含义、迭代对比与分题结果见 **[evaluation/README.md](evaluation/README.md)**。

---

## 示例问题

| 预期路由 | 问题示例 |
|----------|----------|
| vector | 2025 年公司营业收入和扣非净利润分别是多少？ |
| graph | 公司有多少家全资子公司？实际控制人是谁？ |
| hybrid | 招股书列举了哪些与发行人相关的风险？至少四项。 |

---

## 项目结构

```text
config/              配置与 Prompt
src/
  pipeline/          离线：分块、抽取、建库
  agents/            在线：Router、Experts、Generator、Critic
  tools/             Chroma、Neo4j、检索与上下文截断
  graph_workflow.py  LangGraph 编排
evaluation/          测试集、跑题、打分（见 evaluation/README.md）
app.py               CLI
web_app.py           Streamlit
docs/                架构说明
```

---

## 配置

| 变量 | 说明 |
|------|------|
| `NEO4J_*` | 图谱连接 |
| `OLLAMA_LLM_MODEL` | 生成模型，建议 `qwen2.5:7b` |
| `OLLAMA_EMBED_MODEL` | 向量模型，默认 `nomic-embed-text` |
| `RETRIEVAL_TOP_K` | 向量 Top-K |
| `CONTEXT_MAX_CHUNKS` | 送入 LLM 的最大 chunk 数 |
| `CRITIC_USE_LLM` | 默认 `false` |

完整列表见 [.env.example](.env.example)。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [evaluation/README.md](evaluation/README.md) | 测试集、跑分流程、**指标迭代与对比** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 模块职责与数据流 |

---

## License

按仓库实际许可证填写（如 MIT）。

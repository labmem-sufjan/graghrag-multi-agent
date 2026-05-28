# 项目结构与模块说明

## 目录树

```text
graghrag-multi-agent/
├── app.py                      # CLI 问答入口
├── web_app.py                  # Streamlit Web 入口
├── config/
│   ├── settings.py             # 环境变量与全局配置
│   └── prompts.py              # 抽取 / 路由 / 回答 / Critic 提示词
├── data/
│   ├── raw_docs/               # 原始 PDF
│   ├── chroma_db/              # Chroma 持久化
│   └── neo4j/                  # Neo4j 数据卷
├── src/
│   ├── pipeline/               # 离线建库
│   ├── agents/                 # 在线 Multi-Agent 节点
│   ├── tools/                  # 存储与检索客户端
│   └── graph_workflow.py       # LangGraph 编排
├── evaluation/                 # Ragas 评测（待完善）
├── scripts/inspect_knowledge.py
└── docs/
```

## 数据流

```text
[离线] PDF → document_processor → extractor → Neo4j + Chroma
[在线] 问题 → router → expert → generator → critic → 答案
```

## 文件职责

见 README 或面试前阅读本文档各节标题。

"""Streamlit web UI — light unified layout."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="GraphRAG · 招股书智能分析",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, 'PingFang SC', sans-serif; }
    .stApp { background: #f4f6f9; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8ecf1; }
    .app-header {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 1.25rem 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    }
    .app-title { font-size: 1.35rem; font-weight: 600; color: #0f172a; margin: 0 0 .25rem 0; }
    .app-desc { font-size: 0.875rem; color: #64748b; margin: 0; }
    .metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: 1rem; }
    .metric {
        flex: 1; min-width: 100px; background: #fff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: .6rem .85rem; text-align: center;
    }
    .metric b { display: block; font-size: 1.25rem; color: #2563eb; font-weight: 600; }
    .metric span { font-size: .7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .03em; }
    div[data-testid="stChatMessage"] {
        background: #fff !important; border: 1px solid #e8ecf1 !important;
        border-radius: 12px !important; box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }
    .meta-bar {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: .5rem .75rem; margin-top: .5rem; font-size: .8rem; color: #475569;
    }
    .pill {
        display: inline-block; padding: .15rem .5rem; border-radius: 6px;
        font-size: .72rem; font-weight: 500; margin-right: .35rem;
    }
    .pill-vector { background: #dbeafe; color: #1d4ed8; }
    .pill-graph { background: #d1fae5; color: #047857; }
    .pill-hybrid { background: #ffedd5; color: #c2410c; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 820px; padding-top: 1.5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = [
    "宇树科技的主营业务是什么？",
    "公司实际控制人是谁？有哪些全资子公司？",
    "2025年营业收入和扣非净利润是多少？",
    "公司面临的主要风险因素有哪些？",
]


@st.cache_data(ttl=120)
def _kb_stats() -> dict[str, str]:
    stats = {k: "—" for k in ("chunks", "entities", "relations", "chroma")}
    try:
        from src.tools.neo4j_client import Neo4jGraphClient

        with Neo4jGraphClient() as g:
            s = g.stats()
        stats["chunks"] = str(s.get("chunks", 0))
        stats["entities"] = str(s.get("entities", 0))
        stats["relations"] = str(s.get("relations", 0))
    except Exception:
        pass
    try:
        import chromadb
        from config.settings import CHROMA_DIR, settings

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        stats["chroma"] = str(client.get_collection(settings.chroma_collection).count())
    except Exception:
        pass
    return stats


def _pill(route: str) -> str:
    cls = {"graph": "pill-graph", "hybrid": "pill-hybrid"}.get(route, "pill-vector")
    return f'<span class="pill {cls}">{route}</span>'


def _render_meta(meta: dict) -> None:
    route = meta.get("route") or "vector"
    ok = meta.get("critic_passed", True)
    st.markdown(
        f'<div class="meta-bar">'
        f'{_pill(route)} '
        f'<span>{meta.get("route_reason", "")}</span> · '
        f'Critic: {"通过" if ok else "未通过"} — {meta.get("critic_feedback", "")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    ids = meta.get("chunk_ids") or []
    if ids:
        st.caption("引用 chunk: " + " · ".join(ids[:5]) + (" …" if len(ids) > 5 else ""))


def _run_question(question: str) -> dict:
    from src.graph_workflow import run_query

    return run_query(question)


# —— 统一顶栏 ——
kb = _kb_stats()
st.markdown(
    f"""
    <div class="app-header">
        <p class="app-title">招股书智能分析 · GraphRAG Multi-Agent</p>
        <p class="app-desc">宇树科技私有域 · Chroma 向量 + Neo4j 图谱 · 本地 Ollama · LangGraph 编排</p>
        <div class="metric-row">
            <div class="metric"><b>{kb['chunks']}</b><span>Chunks</span></div>
            <div class="metric"><b>{kb['chroma']}</b><span>Vectors</span></div>
            <div class="metric"><b>{kb['entities']}</b><span>Entities</span></div>
            <div class="metric"><b>{kb['relations']}</b><span>Relations</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 示例问题（横向 chips，避免侧边栏分裂感）
st.caption("快速提问")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = ex

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"])

prompt = st.chat_input("输入关于招股书的问题…")
if st.session_state.get("pending_question"):
    prompt = st.session_state.pop("pending_question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答…"):
            try:
                result = _run_question(prompt)
                answer = result.get("answer", "未能生成回答。")
                st.markdown(answer)
                meta = {
                    "route": result.get("route"),
                    "route_reason": result.get("route_reason"),
                    "chunk_ids": result.get("chunk_ids", []),
                    "critic_passed": result.get("critic_passed"),
                    "critic_feedback": result.get("critic_feedback"),
                }
                _render_meta(meta)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )
            except Exception as e:
                err = f"**运行出错**：{e}\n\n请确认 Neo4j、Ollama 已启动，且已完成建库。"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
elif not st.session_state.messages:
    st.info("👆 选择上方示例，或在下方输入框提问。")

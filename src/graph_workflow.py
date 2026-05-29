"""在线问答：LangGraph 编排 Multi-Agent 流水线。

流程：Router →（Vector | Graph | Hybrid 三选一）→ Generate → Critic → END
当前 Critic 不通过也不会回环重试，仅把 critic_passed 写入状态。
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.critic import critique_answer
from src.agents.generator import generate_answer
from src.agents.graph_expert import retrieve_graph
from src.agents.hybrid import retrieve_hybrid
from src.agents.router import route_question
from src.agents.state import AgentState
from src.agents.vector_expert import retrieve_vector


def _pick_retriever(state: AgentState) -> str:
    """条件边：根据 state['route'] 进入对应 Expert 节点。"""
    return state.get("route", "vector")


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("vector_expert", retrieve_vector)
    graph.add_node("graph_expert", retrieve_graph)
    graph.add_node("hybrid_expert", retrieve_hybrid)
    graph.add_node("generate", generate_answer)
    graph.add_node("critic", critique_answer)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _pick_retriever,
        {
            "vector": "vector_expert",
            "graph": "graph_expert",
            "hybrid": "hybrid_expert",
        },
    )
    # 三条 Expert 边都汇入同一 Generate，再进 Critic
    graph.add_edge("vector_expert", "generate")
    graph.add_edge("graph_expert", "generate")
    graph.add_edge("hybrid_expert", "generate")
    graph.add_edge("generate", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def run_query(question: str) -> AgentState:
    """对外入口：app.py / web_app / evaluation multi_agent 均调用此函数。"""
    app = build_workflow()
    initial: AgentState = {
        "question": question,
        "retry_count": 0,
        "chunk_ids": [],
    }
    return app.invoke(initial)

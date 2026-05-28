"""LangGraph multi-agent workflow: Router → Experts → Generate → Critic."""

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
    graph.add_edge("vector_expert", "generate")
    graph.add_edge("graph_expert", "generate")
    graph.add_edge("hybrid_expert", "generate")
    graph.add_edge("generate", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def run_query(question: str) -> AgentState:
    app = build_workflow()
    initial: AgentState = {
        "question": question,
        "retry_count": 0,
        "chunk_ids": [],
    }
    return app.invoke(initial)

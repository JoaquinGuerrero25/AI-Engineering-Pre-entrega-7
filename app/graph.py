"""Orquestador multi-agente del M6 + checkpointer de Redis + nodo de aprobación humana.

Supervisor -> {Investigador, Analista} -> Validación -> [Aprobación humana] -> Síntesis -> END
"""

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents import build_analyst_agent, build_research_agent
from app.agents.common import make_specialist_node
from app.hitl import aprobacion_humana, ejecutar_accion, route_after_approval
from app.state import (
    ANALISTA,
    APROBACION,
    EJECUTAR_ACCION,
    INVESTIGADOR,
    SINTETIZADOR,
    SUPERVISOR,
    VALIDACION,
    OrchestratorState,
)
from app.supervisor import make_supervisor_node, route_after_supervisor
from app.validation import make_synthesizer_node, route_after_validation, validacion


def _contexto(state: OrchestratorState) -> str:
    # Los especialistas solo reciben las notas del investigador, no el historial del sistema.
    return state.get("research_notes", "")


def build_graph(
    llm: BaseChatModel,
    research_llm: BaseChatModel | None = None,
    analyst_llm: BaseChatModel | None = None,
    checkpointer=None,
):
    """Compila el grafo. `checkpointer` (AsyncRedisSaver en producción) persiste cada paso por thread_id."""
    g = StateGraph(OrchestratorState)

    g.add_node(SUPERVISOR, make_supervisor_node(llm))
    g.add_node(INVESTIGADOR, make_specialist_node(INVESTIGADOR, build_research_agent(research_llm or llm), "research_notes", _contexto))
    g.add_node(ANALISTA, make_specialist_node(ANALISTA, build_analyst_agent(analyst_llm or llm), "analysis_result", _contexto))
    g.add_node(VALIDACION, validacion)
    g.add_node(APROBACION, aprobacion_humana)
    g.add_node(EJECUTAR_ACCION, ejecutar_accion)
    g.add_node(SINTETIZADOR, make_synthesizer_node(llm))

    g.add_edge(START, SUPERVISOR)
    g.add_conditional_edges(SUPERVISOR, route_after_supervisor)
    g.add_edge(INVESTIGADOR, SUPERVISOR)
    g.add_edge(ANALISTA, SUPERVISOR)
    # Validación aprobada (o sin presupuesto) -> pasa por la compuerta HITL antes de sintetizar.
    g.add_conditional_edges(VALIDACION, route_after_validation, {SUPERVISOR: SUPERVISOR, SINTETIZADOR: APROBACION})
    g.add_conditional_edges(
        APROBACION, route_after_approval, {EJECUTAR_ACCION: EJECUTAR_ACCION, SINTETIZADOR: SINTETIZADOR, END: END}
    )
    g.add_edge(EJECUTAR_ACCION, SINTETIZADOR)
    g.add_edge(SINTETIZADOR, END)

    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def _llm_real() -> BaseChatModel:
    from app.llm import crear_llm

    return crear_llm()


def crear_grafo(checkpointer=None):
    """Grafo listo para ejecutar. LLM_PROVIDER=mock usa LLMs guionados (sin API keys, uno nuevo por corrida)."""
    load_dotenv_once()
    if os.getenv("LLM_PROVIDER", "").lower() == "mock":
        from app.fakes import modelos_demo

        sup, inv, ana = modelos_demo()
        return build_graph(sup, inv, ana, checkpointer=checkpointer)
    return build_graph(_llm_real(), checkpointer=checkpointer)


def load_dotenv_once() -> None:
    from dotenv import load_dotenv

    load_dotenv()

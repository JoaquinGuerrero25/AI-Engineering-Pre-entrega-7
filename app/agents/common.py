"""Utilidades compartidas: nodo especialista con contexto aislado."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.state import OrchestratorState


def texto(content: Any) -> str:
    """Normaliza el contenido de un mensaje (str o lista de bloques) a texto plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = [b if isinstance(b, str) else b.get("text", "") for b in content if b]
        return "".join(partes)
    return str(content)


def make_specialist_node(
    name: str,
    agent: Any,
    result_field: str,
    context: Callable[[OrchestratorState], str],
) -> Callable[[OrchestratorState], dict]:
    """Envuelve un agente ReAct como nodo del grafo.

    Evita la contaminación de contexto: el especialista NO ve `messages` del
    sistema; recibe solo la instrucción del supervisor + el contexto mínimo que
    `context(state)` decida compartirle.
    """

    def node(state: OrchestratorState) -> dict:
        instruccion = state["instruction"]
        extra = context(state)
        prompt = f"Tarea: {instruccion}"
        if extra:
            prompt += f"\n\nContexto disponible:\n{extra}"

        resultado = agent.invoke({"messages": [HumanMessage(prompt)]}, {"recursion_limit": 14})
        mensajes = resultado["messages"]
        salida = next((texto(m.content) for m in reversed(mensajes) if isinstance(m, AIMessage) and m.content), "")
        herramientas = [m.name for m in mensajes if isinstance(m, ToolMessage) and m.name]

        return {
            "messages": [AIMessage(content=salida, name=name)],
            result_field: salida,
            "contributions": [
                {
                    "agent": name,
                    "step": state.get("step_count", 0),
                    "instruction": instruccion,
                    "output": salida,
                    "tools_used": herramientas,
                }
            ],
        }

    return node

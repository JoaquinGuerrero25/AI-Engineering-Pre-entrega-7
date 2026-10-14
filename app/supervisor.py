"""Nodo Supervisor: router inteligente + controlador de flujo.

El LLM propone una `Decision` estructurada; luego `aplicar_guardas` la corrige con
reglas deterministas (tope de pasos, orden lógico, límite de re-delegaciones) para que
el flujo NUNCA dependa solo del criterio del modelo.
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.common import texto
from app.state import (
    ANALISTA,
    FINISH,
    INVESTIGADOR,
    MAX_REVISIONS,
    MAX_STEPS,
    VALIDACION,
    OrchestratorState,
)

SUPERVISOR_PROMPT = """Eres el Supervisor de un equipo con dos especialistas:
- investigador: recupera datos de la base de conocimiento (reseñas, ventas, políticas). No calcula.
- analista: procesa datos YA recuperados (sentimiento, estadística, cálculos, validación). No busca.

Dada la conversación actual, decide quién debe intervenir ahora o si es momento de finalizar (FINISH).

RÚBRICA DE SUFICIENCIA (solo puedes elegir FINISH si se cumplen TODOS los puntos):
1. El investigador entregó datos concretos y citados con [doc-XXX] que cubren lo que pidió el usuario.
2. El analista procesó esos datos con herramientas y entregó resultados numéricos explícitos.
3. Cada parte de la pregunta del usuario tiene una respuesta respaldada por esos resultados.
Si falta algo, delega al especialista correspondiente con una `instruction` concreta, autocontenida y
específica (qué buscar / qué calcular). Si un especialista ya entregó pero su salida es incompleta o
errónea, puedes pedirle UNA revisión indicando exactamente qué corregir.
Orden habitual: investigador -> analista -> FINISH. El analista nunca va antes de tener datos.
Si hay "Feedback de validación" pendiente, corrígelo antes de finalizar."""


class Decision(BaseModel):
    next_agent: Literal["investigador", "analista", "FINISH"] = Field(
        description="Especialista que debe actuar ahora, o FINISH si la rúbrica se cumple."
    )
    instruction: str = Field(
        default="", description="Tarea concreta y autocontenida para el especialista (vacía si FINISH)."
    )
    reason: str = Field(description="Una frase: por qué esta decisión según la rúbrica.")


def _resumen_estado(state: OrchestratorState) -> str:
    """Resumen compacto para el supervisor (no se le pasa todo el historial crudo)."""
    pregunta = next((texto(m.content) for m in state["messages"] if isinstance(m, HumanMessage)), "")
    lineas = [f"Pregunta del usuario: {pregunta}", f"Pasos usados: {state.get('step_count', 0)}/{MAX_STEPS}"]
    if not state.get("contributions"):
        lineas.append("Aún no contribuyó ningún especialista.")
    for c in state.get("contributions", []):
        salida = c["output"] if len(c["output"]) < 1500 else c["output"][:1500] + "…[recortado]"
        lineas.append(
            f"--- Aporte de {c['agent']} (paso {c['step']}, herramientas: {c['tools_used'] or 'ninguna'}) ---\n{salida}"
        )
    if state.get("validation_feedback"):
        lineas.append(f"Feedback de validación pendiente: {state['validation_feedback']}")
    return "\n".join(lineas)


def aplicar_guardas(decision: Decision, state: OrchestratorState) -> tuple[str, str, str]:
    """Corrige la decisión del LLM con reglas deterministas. Devuelve (siguiente, instrucción, nota)."""
    siguiente, instruccion = decision.next_agent, decision.instruction
    revisiones = state.get("revisions", {})
    ya_aportaron = {c["agent"] for c in state.get("contributions", [])}

    if state.get("step_count", 0) >= MAX_STEPS:
        return FINISH, "", f"Tope de {MAX_STEPS} pasos alcanzado: se fuerza el cierre."

    if siguiente == ANALISTA and not state.get("research_notes"):
        return INVESTIGADOR, instruccion, "El analista no puede actuar sin datos: se redirige al investigador."

    if siguiente in (INVESTIGADOR, ANALISTA) and not instruccion.strip():
        instruccion = f"Resuelve la parte de tu especialidad de la pregunta del usuario."

    if siguiente in ya_aportaron and revisiones.get(siguiente, 0) >= MAX_REVISIONS:
        pendientes = [a for a in (INVESTIGADOR, ANALISTA) if a not in ya_aportaron]
        if pendientes:
            return pendientes[0], instruccion, f"{siguiente} agotó sus revisiones: se delega a {pendientes[0]}."
        return FINISH, "", f"{siguiente} agotó sus revisiones ({MAX_REVISIONS}): se cierra con lo disponible."

    return siguiente, instruccion, ""


def make_supervisor_node(llm: BaseChatModel):
    decisor = llm.with_structured_output(Decision)

    def supervisor(state: OrchestratorState) -> dict:
        decision: Decision = decisor.invoke(
            [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(_resumen_estado(state))]
        )
        siguiente, instruccion, nota = aplicar_guardas(decision, state)

        revisiones = dict(state.get("revisions", {}))
        if siguiente in {c["agent"] for c in state.get("contributions", [])}:
            revisiones[siguiente] = revisiones.get(siguiente, 0) + 1

        motivo = decision.reason + (f" [guarda: {nota}]" if nota else "")
        destino = "validación final" if siguiente == FINISH else siguiente
        return {
            "next_agent": siguiente,
            "instruction": instruccion,
            "step_count": state.get("step_count", 0) + 1,
            "revisions": revisiones,
            "messages": [AIMessage(content=f"Decisión → {destino}. {motivo}", name="supervisor")],
        }

    return supervisor


def route_after_supervisor(state: OrchestratorState) -> Literal["investigador", "analista", "validacion"]:
    """Arista condicional: mapea la decisión del supervisor al nombre de un nodo."""
    siguiente = state["next_agent"]
    if siguiente == INVESTIGADOR:
        return INVESTIGADOR
    if siguiente == ANALISTA:
        return ANALISTA
    return VALIDACION

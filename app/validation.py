"""Nodos de Validación (rúbrica determinista) y Síntesis final."""

import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.common import texto
from app.state import ANALISTA, INVESTIGADOR, MAX_STEPS, SINTETIZADOR, SUPERVISOR, OrchestratorState

MAX_VALIDATION_ATTEMPTS = 2

SYNTH_PROMPT = """Eres el redactor final. Responde la pregunta del usuario en español usando SOLO los
resultados del investigador y del analista que se te entregan. Reglas:
- Cita las fuentes [doc-XXX] junto a los datos.
- Reporta las cifras tal cual (no recalcules).
- Si hay "Advertencias", menciónalas al final como limitaciones.
Formato: respuesta directa, breve, con las cifras clave."""


def revisar_rubrica(state: OrchestratorState) -> list[str]:
    """Rúbrica de aceptación antes del END. Devuelve la lista de incumplimientos."""
    problemas = []
    notas = state.get("research_notes", "")
    analisis = state.get("analysis_result", "")
    contribuciones = state.get("contributions", [])

    if not notas.strip():
        problemas.append(f"{INVESTIGADOR}: no entregó resultados de investigación.")
    elif not re.search(r"\[?doc-\d{3}\]?", notas):
        problemas.append(f"{INVESTIGADOR}: sus datos no citan ninguna fuente [doc-XXX].")

    if not analisis.strip():
        problemas.append(f"{ANALISTA}: no entregó análisis.")
    else:
        if not any(c["agent"] == ANALISTA and c["tools_used"] for c in contribuciones):
            problemas.append(f"{ANALISTA}: no usó herramientas de cómputo (cifras sin respaldo).")
        if not re.search(r"\d", analisis):
            problemas.append(f"{ANALISTA}: el análisis no contiene resultados numéricos.")
    return problemas


def validacion(state: OrchestratorState) -> dict:
    problemas = revisar_rubrica(state)
    intentos = state.get("validation_attempts", 0) + 1
    if not problemas:
        return {"validation_feedback": "", "validation_attempts": intentos,
                "messages": [AIMessage(content="Validación OK: rúbrica cumplida.", name="validacion")]}  # fmt: skip
    feedback = " | ".join(problemas)
    return {
        "validation_feedback": feedback,
        "validation_attempts": intentos,
        "messages": [AIMessage(content=f"Validación RECHAZADA: {feedback}", name="validacion")],
    }


def route_after_validation(state: OrchestratorState) -> Literal["supervisor", "sintetizador"]:
    """Si la rúbrica falla y aún hay presupuesto, vuelve al supervisor; si no, cierra con advertencias."""
    hay_presupuesto = (
        state.get("validation_attempts", 0) < MAX_VALIDATION_ATTEMPTS and state.get("step_count", 0) < MAX_STEPS
    )
    if state.get("validation_feedback") and hay_presupuesto:
        return SUPERVISOR
    return SINTETIZADOR


def make_synthesizer_node(llm: BaseChatModel):
    def sintetizador(state: OrchestratorState) -> dict:
        pregunta = next((texto(m.content) for m in state["messages"] if isinstance(m, HumanMessage)), "")
        partes = [
            f"Pregunta: {pregunta}",
            f"Resultados del investigador:\n{state.get('research_notes') or '(sin datos)'}",
            f"Resultados del analista:\n{state.get('analysis_result') or '(sin análisis)'}",
        ]
        if state.get("validation_feedback"):
            partes.append(f"Advertencias (la rúbrica no se cumplió del todo): {state['validation_feedback']}")
        respuesta = texto(llm.invoke([SystemMessage(SYNTH_PROMPT), HumanMessage("\n\n".join(partes))]).content)
        return {
            "final_answer": respuesta,
            "task_completed": True,
            "messages": [AIMessage(content=respuesta, name="sintetizador")],
        }

    return sintetizador

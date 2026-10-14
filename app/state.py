"""Estado compartido del orquestador.

Hereda de MessagesState (campo `messages`) y agrega campos estructurados para que
cada contribución quede atribuida a un agente y no se pierda contexto entre turnos.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState

# Nombres de los nodos del grafo (única fuente de verdad para el routing).
SUPERVISOR = "supervisor"
INVESTIGADOR = "investigador"
ANALISTA = "analista"
VALIDACION = "validacion"
SINTETIZADOR = "sintetizador"
APROBACION = "aprobacion_humana"
EJECUTAR_ACCION = "ejecutar_accion"
FINISH = "FINISH"

MAX_STEPS = 6  # tope de decisiones del supervisor: corta el "Supervisor Infinito"
MAX_REVISIONS = 1  # cuántas veces se puede re-delegar a un mismo especialista


class Contribution(TypedDict):
    """Registro de lo que aportó un agente: quién, en qué paso, con qué herramientas."""

    agent: str
    step: int
    instruction: str
    output: str
    tools_used: list[str]


class OrchestratorState(MessagesState):
    # --- control de flujo ---
    next_agent: str  # decisión del supervisor: investigador | analista | FINISH
    instruction: str  # tarea concreta para el especialista elegido (lo único que recibe)
    step_count: int  # decisiones tomadas por el supervisor
    revisions: dict[str, int]  # re-delegaciones por agente
    task_completed: bool

    # --- resultados por agente (trazabilidad) ---
    research_notes: str
    analysis_result: str
    contributions: Annotated[list[Contribution], operator.add]  # log append-only

    # --- validación y cierre ---
    validation_feedback: str  # motivo del último rechazo de la rúbrica
    validation_attempts: int
    final_answer: str

    # --- human-in-the-loop ---
    force_approval: bool  # el cliente pidió explícitamente aprobación humana
    requires_approval: bool  # el sistema clasificó la tarea como crítica
    approval: dict  # decisión humana: {"approved": bool, "reviewer": str, "comment": str}
    action_log: Annotated[list[str], operator.add]  # efectos secundarios ejecutados (simulados)

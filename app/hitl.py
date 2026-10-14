"""Human-in-the-loop: nodo de aprobación (interrupt) + endpoint POST /tasks/{id}/approve.

Una tarea es "crítica" si pide una acción con efectos secundarios (enviar, reembolsar,
eliminar...) o si el cliente fuerza la revisión. En ese caso el grafo se PAUSA con
`interrupt()` —el estado queda guardado en el checkpointer de Redis— y no avanza hasta que
llegue una señal externa (`Command(resume=...)`) desde el endpoint de aprobación.
"""

import re
import unicodedata
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import BaseModel

from app.agents.common import texto
from app.state import EJECUTAR_ACCION, SINTETIZADOR, OrchestratorState

# Raíces de verbos con efectos secundarios (comparadas sin tildes, en minúsculas).
ACCIONES_CRITICAS = (
    "envia", "enviar", "mand", "emit", "reembols", "elimin", "borr", "cancel",
    "pagar", "paga ", "transfer", "publica", "ejecut", "despleg", "comprar", "notifica",
)  # fmt: skip


def _normalizar(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def clasificar_criticidad(consulta: str, forzada: bool = False) -> str:
    """Devuelve el motivo por el que la tarea es crítica, o "" si no lo es."""
    if forzada:
        return "El cliente solicitó revisión humana explícita."
    q = _normalizar(consulta)
    hallados = sorted({a.strip() for a in ACCIONES_CRITICAS if re.search(rf"\b{re.escape(a.strip())}", q)})
    if hallados:
        return f"La tarea implica una acción con efectos secundarios (detectado: {', '.join(hallados)})."
    return ""


def _pregunta(state: OrchestratorState) -> str:
    return next((texto(m.content) for m in state["messages"] if isinstance(m, HumanMessage)), "")


def aprobacion_humana(state: OrchestratorState) -> dict:
    """Nodo HITL. Si la tarea es crítica, pausa el grafo hasta recibir la decisión humana."""
    motivo = clasificar_criticidad(_pregunta(state), state.get("force_approval", False))
    if not motivo:
        return {"requires_approval": False}

    # --- PAUSA: lo anterior a interrupt() se re-ejecuta al reanudar, por eso es idempotente ---
    decision = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "motivo": motivo,
            "pregunta": _pregunta(state),
            "resumen_analisis": state.get("analysis_result", "")[:1200],
        }
    )
    aprobada = bool(decision.get("approved"))
    registro = {
        "approved": aprobada,
        "reviewer": decision.get("reviewer", "anonimo"),
        "comment": decision.get("comment", ""),
    }
    if aprobada:
        msg = f"Aprobada por {registro['reviewer']}."
        return {"requires_approval": True, "approval": registro, "messages": [AIMessage(content=msg, name="aprobacion")]}
    msg = f"Tarea RECHAZADA por {registro['reviewer']}. Motivo: {registro['comment'] or 'sin comentario'}. No se ejecutó ninguna acción."
    return {
        "requires_approval": True,
        "approval": registro,
        "final_answer": msg,
        "task_completed": True,
        "messages": [AIMessage(content=msg, name="aprobacion")],
    }


def route_after_approval(state: OrchestratorState) -> Literal["ejecutar_accion", "sintetizador", "__end__"]:
    if not state.get("requires_approval"):
        return SINTETIZADOR
    return EJECUTAR_ACCION if state["approval"]["approved"] else END


def ejecutar_accion(state: OrchestratorState) -> dict:
    """Efecto secundario (simulado) que solo ocurre tras la aprobación humana."""
    entrada = f"Acción ejecutada tras aprobación de {state['approval']['reviewer']}: {_pregunta(state)[:200]}"
    return {"action_log": [entrada]}


# ---------------------------------------------------------------- endpoint
router = APIRouter()


class ApprovalIn(BaseModel):
    approved: bool
    reviewer: str = "anonimo"
    comment: str = ""


@router.post("/tasks/{job_id}/approve", status_code=202)
async def approve(job_id: str, body: ApprovalIn, request: Request) -> dict:
    """Señal externa que reanuda un job pausado en WAITING_APPROVAL."""
    jobs = request.app.state.jobs
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job inexistente")
    if job["status"] != "WAITING_APPROVAL":
        raise HTTPException(409, f"El job está en estado {job['status']}; solo se aprueba WAITING_APPROVAL.")
    if not await jobs.claim_approval(job_id):  # evita doble aprobación concurrente
        raise HTTPException(409, "El job ya fue resuelto.")
    await jobs.update(job_id, status="PENDING", approval=body.model_dump())
    await jobs.enqueue({"job_id": job_id, "resume": body.model_dump()})
    return {"job_id": job_id, "status": "PENDING"}

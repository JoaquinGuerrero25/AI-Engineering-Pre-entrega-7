"""Worker: consume la cola de Redis y ejecuta el grafo en segundo plano.

Ciclo de vida del job: PENDING -> RUNNING -> (WAITING_APPROVAL -> PENDING -> RUNNING)* -> DONE | REJECTED | FAILED.
Cualquier excepción se captura y deja el job en FAILED con el error, para que el cliente nunca
quede esperando en un polling infinito.
"""

import asyncio
import logging
import os
import traceback

from langgraph.types import Command

from app import store
from app.observability import tracer
from app.store import JobStore

log = logging.getLogger(__name__)

JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "600"))


async def procesar(job_id: str, resume: dict | None, jobs: JobStore, crear_grafo) -> None:
    job = await jobs.get(job_id)
    if job is None:
        return
    await jobs.update(job_id, status=store.RUNNING)
    config = {"configurable": {"thread_id": job_id}, "recursion_limit": 40}

    # Span raíz: agrupa las trazas del job y las asocia a una "sesión" de Phoenix.
    with tracer().start_as_current_span("job") as span:
        span.set_attribute("session.id", job_id)
        span.set_attribute("job.id", job_id)
        span.set_attribute("job.resume", resume is not None)
        try:
            grafo = crear_grafo()
            if resume is not None:
                entrada = Command(resume=resume)
            else:
                entrada = {"messages": [("user", job["query"])], "force_approval": bool(job.get("force_approval"))}
            # ainvoke: los nodos síncronos (LLM) corren en el threadpool de LangGraph, sin bloquear el loop.
            resultado = await asyncio.wait_for(grafo.ainvoke(entrada, config), timeout=JOB_TIMEOUT_S)

            interrupciones = resultado.get("__interrupt__")
            if interrupciones:
                await jobs.update(job_id, status=store.WAITING_APPROVAL, approval_request=interrupciones[0].value)
            else:
                aprobado = resultado.get("approval", {}).get("approved", True)
                await jobs.update(
                    job_id,
                    status=store.DONE if aprobado else store.REJECTED,
                    result=resultado.get("final_answer", ""),
                    action_log=resultado.get("action_log", []),
                )
        except Exception as e:  # noqa: BLE001 - toda falla debe reflejarse en Redis
            log.error("Job %s falló: %s", job_id, traceback.format_exc())
            span.record_exception(e)
            msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            try:
                await jobs.update(job_id, status=store.FAILED, error=msg)
            except Exception:  # noqa: BLE001
                log.exception("No se pudo marcar FAILED el job %s", job_id)


async def worker_loop(nombre: str, jobs: JobStore, crear_grafo) -> None:
    log.info("worker %s listo", nombre)
    while True:
        try:
            item = await jobs.dequeue(timeout=1)
            if item:
                await procesar(item["job_id"], item.get("resume"), jobs, crear_grafo)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - un error de Redis no mata al worker
            log.exception("error en el loop del worker %s", nombre)
            await asyncio.sleep(1)

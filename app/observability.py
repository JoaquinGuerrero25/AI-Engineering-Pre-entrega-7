"""Observabilidad con Arize Phoenix (OpenInference).

`init_observability()` registra un exportador OTLP hacia Phoenix e instrumenta LangChain/LangGraph:
cada nodo del grafo, cada llamada al LLM (con tokens de entrada/salida y modelo, de donde Phoenix
calcula el costo) y cada herramienta queda como un span dentro de la traza del job.
"""

import logging
import os

from opentelemetry import trace

log = logging.getLogger(__name__)
_iniciado = False


def init_observability() -> None:
    global _iniciado
    if _iniciado or os.getenv("PHOENIX_DISABLED") == "1":
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "agente-multiagente-prod"),
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        batch=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _iniciado = True
    log.info("Phoenix activo -> %s", endpoint)


def tracer() -> trace.Tracer:
    return trace.get_tracer("agente-api")

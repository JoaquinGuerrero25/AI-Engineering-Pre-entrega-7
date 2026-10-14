"""LLM guionado (sin red ni API keys) para tests y para `main.py --mock`.

Cada modelo reproduce en orden las respuestas que se le cargan. Las llamadas a herramientas
(tool_calls) se ejecutan de verdad: solo el "razonamiento" del LLM está simulado.
"""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agents.analyst_agent import analizar_sentimiento, calcular_estadisticas
from app.agents.research_agent import _cargar_kb
from app.supervisor import Decision


class ScriptedChat(BaseChatModel):
    replies: list[AIMessage] = []  # respuestas de chat / tool-calling, en orden
    decisions: list[Decision] = []  # respuestas de with_structured_output, en orden
    _i: int = 0
    _j: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any) -> ChatResult:
        msg = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Sequence[Any], **kw: Any):
        return self

    def with_structured_output(self, schema: Any, **kw: Any):
        def _next(_: Any) -> Decision:
            d = self.decisions[min(self._j, len(self.decisions) - 1)]
            self._j += 1
            return d

        return RunnableLambda(_next)


def tool_call(name: str, args: dict, id_: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def modelos_demo() -> tuple[ScriptedChat, ScriptedChat, ScriptedChat]:
    """Guion del flujo Supervisor -> Investigador -> Analista -> Validación -> Síntesis."""
    resenas = [d for d in _cargar_kb() if d["tema"] == "reseñas AuroraBook X14"]
    textos = [d["texto"] for d in resenas]
    califs = [d["calificacion"] for d in resenas]
    notas = "Reseñas de la AuroraBook X14:\n" + "\n".join(
        f"- [{d['id']}] ({d['calificacion']}/5) {d['texto']}" for d in resenas
    )

    sent = json.loads(analizar_sentimiento.invoke({"textos": textos}))
    stats = json.loads(calcular_estadisticas.invoke({"valores": califs}))
    conteo = sent["conteo"]
    resumen = (
        f"Análisis de [doc-001] a [doc-005]: calificación media {stats['media']}/5 (mediana {stats['mediana']}, "
        f"mín {stats['min']}, máx {stats['max']}, desvío {stats['desvio']}); sentimiento global {sent['etiqueta_global']} "
        f"(puntaje medio {sent['puntaje_medio']}): {conteo['positivo']} positivas, {conteo['neutral']} neutrales, "
        f"{conteo['negativo']} negativas."
    )

    supervisor_llm = ScriptedChat(
        decisions=[
            Decision(next_agent="investigador", instruction="Recupera todas las reseñas de la AuroraBook X14 con su calificación.", reason="Faltan datos: nada fue investigado todavía."),
            Decision(next_agent="analista", instruction="Analiza el sentimiento de las reseñas y calcula la calificación promedio.", reason="Hay datos citados; falta el análisis."),
            Decision(next_agent="FINISH", reason="La rúbrica se cumple: datos citados y resultados numéricos."),
        ],
        replies=[AIMessage(content=f"La AuroraBook X14 tiene una calificación media de {stats['media']}/5 con sentimiento {sent['etiqueta_global']} [doc-001..005]. {resumen}")],
    )  # fmt: skip
    research_llm = ScriptedChat(
        replies=[
            tool_call("buscar_en_base_conocimiento", {"consulta": "reseñas AuroraBook X14"}, "c1"),
            AIMessage(content=notas),
        ]
    )
    analyst_llm = ScriptedChat(
        replies=[
            tool_call("analizar_sentimiento", {"textos": textos}, "c2"),
            tool_call("calcular_estadisticas", {"valores": califs}, "c3"),
            AIMessage(content=resumen),
        ]
    )  # fmt: skip
    return supervisor_llm, research_llm, analyst_llm


__all__ = ["ScriptedChat", "tool_call", "modelos_demo", "json"]

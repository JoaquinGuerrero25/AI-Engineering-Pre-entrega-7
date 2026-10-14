"""Agente de Investigación: consulta una base de conocimiento simulada (JSON local)."""

import json
import unicodedata
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "base_conocimiento.json"

SYSTEM_PROMPT = """Eres el Agente de Investigación. Tu único trabajo es RECUPERAR información.
- Usa `buscar_en_base_conocimiento` (puedes llamarla varias veces con consultas distintas) y
  `leer_documento` si necesitas el texto completo.
- No analices ni calcules: eso lo hace otro agente. Entrega los datos crudos y relevantes.
- Responde con una lista de hallazgos; cada dato debe citar su documento como [doc-XXX].
- Incluye los textos completos de reseñas y las cifras exactas (calificaciones, unidades, precios).
- Si no encuentras nada, dilo explícitamente. No inventes datos."""


def _normalizar(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _cargar_kb() -> list[dict]:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def _puntaje(doc: dict, terminos: set[str]) -> int:
    campos = _normalizar(f"{doc['titulo']} {doc['tema']} {doc['texto']}")
    return sum(1 for t in terminos if t in campos)


@tool
def buscar_en_base_conocimiento(consulta: str, max_resultados: int = 5) -> str:
    """Busca documentos relevantes (reseñas, ventas, políticas, docs) por palabras clave.

    Devuelve un JSON con id, título, fuente, calificación (si aplica) y texto completo.
    """
    terminos = {t for t in _normalizar(consulta).split() if len(t) > 2}
    ranking = [(_puntaje(d, terminos), d) for d in _cargar_kb()]
    ranking = sorted((r for r in ranking if r[0] > 0), key=lambda r: -r[0])[:max_resultados]
    if not ranking:
        return json.dumps({"resultados": [], "aviso": "Sin coincidencias."}, ensure_ascii=False)
    return json.dumps(
        {
            "resultados": [
                {k: d[k] for k in ("id", "titulo", "fuente", "calificacion", "texto")} for _, d in ranking
            ]
        },
        ensure_ascii=False,
    )


@tool
def leer_documento(doc_id: str) -> str:
    """Devuelve un documento completo de la base de conocimiento por su id (ej. 'doc-008')."""
    for d in _cargar_kb():
        if d["id"] == doc_id:
            return json.dumps(d, ensure_ascii=False)
    return json.dumps({"error": f"Documento {doc_id!r} no existe."}, ensure_ascii=False)


RESEARCH_TOOLS = [buscar_en_base_conocimiento, leer_documento]


def build_research_agent(llm: BaseChatModel):
    return create_react_agent(llm, RESEARCH_TOOLS, prompt=SYSTEM_PROMPT, name="investigador")

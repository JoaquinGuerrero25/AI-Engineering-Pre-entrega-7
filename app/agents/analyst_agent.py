"""Agente de Análisis/Cómputo: sentimiento, estadística, cálculo seguro y validación de esquemas."""

import ast
import json
import operator
import statistics
import unicodedata

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """Eres el Agente de Análisis. Procesas datos que YA fueron recuperados por el investigador.
- NO tienes acceso a fuentes externas: trabaja solo con el contexto recibido.
- Nunca calcules de cabeza: usa SIEMPRE las herramientas (`analizar_sentimiento`,
  `calcular_estadisticas`, `calcular_expresion`, `validar_esquema`).
- Reporta resultados numéricos exactos tal como los devuelven las herramientas y cita los [doc-XXX] analizados.
- Si el contexto no alcanza para el análisis pedido, dilo explícitamente."""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ---------- sentimiento (léxico simple en español, sin tildes) ----------
_POS = {
    "excelente", "increible", "comodo", "rapido", "recomiendo", "recomendable", "fantastica", "fantastico",
    "buena", "bueno", "buen", "mejor", "encanta", "nitida", "liviana", "genial", "perfecta", "perfecto",
    "correcta", "responde", "totalmente",
}  # fmt: skip
_NEG = {
    "decepcionante", "terrible", "pesimo", "pesima", "lento", "lenta", "calienta", "danada", "danado",
    "tardo", "traba", "regular", "ruido", "problema", "malo", "mala", "peor", "falla",
}  # fmt: skip
_NEGADORES = {"no", "nunca", "tampoco"}


def _puntaje_texto(t: str) -> float:
    tokens = [w.strip(".,;:!¡?¿()\"'") for w in _norm(t).split()]
    pos = neg = 0
    for i, w in enumerate(tokens):
        negado = i > 0 and tokens[i - 1] in _NEGADORES
        if w in _POS:
            pos, neg = (pos, neg + 1) if negado else (pos + 1, neg)
        elif w in _NEG:
            pos, neg = (pos + 1, neg) if negado else (pos, neg + 1)
    total = pos + neg
    return 0.0 if total == 0 else round((pos - neg) / total, 3)


def _etiqueta(p: float) -> str:
    return "positivo" if p > 0.2 else "negativo" if p < -0.2 else "neutral"


@tool
def analizar_sentimiento(textos: list[str]) -> str:
    """Clasifica el sentimiento de cada texto (puntaje -1 a 1) y devuelve el agregado."""
    detalle = []
    for i, t in enumerate(textos, 1):
        p = _puntaje_texto(t)
        detalle.append({"n": i, "puntaje": p, "etiqueta": _etiqueta(p), "inicio": t[:50]})
    conteo = {e: sum(1 for d in detalle if d["etiqueta"] == e) for e in ("positivo", "neutral", "negativo")}
    media = round(statistics.fmean(d["puntaje"] for d in detalle), 3) if detalle else 0.0
    return json.dumps(
        {"detalle": detalle, "conteo": conteo, "puntaje_medio": media, "etiqueta_global": _etiqueta(media)},
        ensure_ascii=False,
    )


# ---------- estadística ----------
@tool
def calcular_estadisticas(valores: list[float]) -> str:
    """Calcula n, media, mediana, mínimo, máximo y desvío estándar de una lista de números."""
    if not valores:
        return json.dumps({"error": "Lista vacía."})
    return json.dumps(
        {
            "n": len(valores),
            "media": round(statistics.fmean(valores), 3),
            "mediana": statistics.median(valores),
            "min": min(valores),
            "max": max(valores),
            "desvio": round(statistics.stdev(valores), 3) if len(valores) > 1 else 0.0,
        }
    )


# ---------- cálculo seguro ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}  # fmt: skip


def _eval(n: ast.AST) -> float:
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 10:
            raise ValueError("Exponente demasiado grande.")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.operand))
    raise ValueError("Expresión no permitida (solo números y + - * / ** %).")


@tool
def calcular_expresion(expresion: str) -> str:
    """Evalúa una expresión aritmética segura, ej. '1200*950 + 1500*950'."""
    try:
        resultado = _eval(ast.parse(expresion, mode="eval").body)
        return json.dumps({"expresion": expresion, "resultado": round(resultado, 4)})
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return json.dumps({"error": str(e)})


# ---------- validación de esquemas ----------
_TIPOS = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


@tool
def validar_esquema(datos_json: str, esquema_json: str) -> str:
    """Valida un JSON contra un esquema mínimo: {"required": [...], "properties": {campo: {"type": ...}}}."""
    try:
        datos, esquema = json.loads(datos_json), json.loads(esquema_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valido": False, "errores": [f"JSON inválido: {e}"]}, ensure_ascii=False)
    errores = [f"Falta el campo requerido '{c}'." for c in esquema.get("required", []) if c not in datos]
    for campo, regla in esquema.get("properties", {}).items():
        tipo = _TIPOS.get(regla.get("type", ""))
        if campo not in datos or tipo is None:
            continue
        valor = datos[campo]
        es_bool = isinstance(valor, bool)
        if not isinstance(valor, tipo) or (es_bool and tipo is not bool):
            errores.append(f"'{campo}' debe ser {regla['type']}.")
    return json.dumps({"valido": not errores, "errores": errores}, ensure_ascii=False)


ANALYST_TOOLS = [analizar_sentimiento, calcular_estadisticas, calcular_expresion, validar_esquema]


def build_analyst_agent(llm: BaseChatModel):
    return create_react_agent(llm, ANALYST_TOOLS, prompt=SYSTEM_PROMPT, name="analista")

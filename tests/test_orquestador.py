"""Tests sin red: herramientas reales + LLMs guionados."""

import json

from langchain_core.messages import AIMessage

from app.agents.analyst_agent import (
    analizar_sentimiento,
    calcular_estadisticas,
    calcular_expresion,
    validar_esquema,
)
from app.agents.research_agent import buscar_en_base_conocimiento, leer_documento
from app.fakes import ScriptedChat, modelos_demo, tool_call
from app.graph import build_graph
from app.state import MAX_STEPS
from app.supervisor import Decision, aplicar_guardas
from app.validation import revisar_rubrica, route_after_validation


def ejecutar(grafo, consulta, verbose=False):
    """Corre el grafo sin checkpointer y devuelve el estado final."""
    return grafo.invoke({"messages": [("user", consulta)]}, {"recursion_limit": 40})


# ---------- herramientas ----------
def test_busqueda_encuentra_resenas_y_acepta_tildes():
    r = json.loads(buscar_en_base_conocimiento.invoke({"consulta": "reseñas aurorabook"}))
    assert {"doc-001", "doc-005"} <= {d["id"] for d in r["resultados"]}
    assert json.loads(buscar_en_base_conocimiento.invoke({"consulta": "xyzzy"}))["resultados"] == []


def test_leer_documento():
    assert json.loads(leer_documento.invoke({"doc_id": "doc-008"}))["id"] == "doc-008"
    assert "error" in json.loads(leer_documento.invoke({"doc_id": "doc-999"}))


def test_sentimiento_y_negacion():
    r = json.loads(analizar_sentimiento.invoke({"textos": ["Excelente y rápido", "Terrible, pésimo servicio", "no recomiendo"]}))
    assert [d["etiqueta"] for d in r["detalle"]] == ["positivo", "negativo", "negativo"]


def test_estadisticas_y_calculo_seguro():
    assert json.loads(calcular_estadisticas.invoke({"valores": [5, 4, 2, 5, 1]}))["media"] == 3.4
    assert json.loads(calcular_expresion.invoke({"expresion": "1200*950 + 1500*950"}))["resultado"] == 2565000
    assert "error" in json.loads(calcular_expresion.invoke({"expresion": "__import__('os').system('x')"}))
    assert "error" in json.loads(calcular_expresion.invoke({"expresion": "9**99"}))


def test_validar_esquema():
    esq = json.dumps({"required": ["a"], "properties": {"a": {"type": "integer"}}})
    assert json.loads(validar_esquema.invoke({"datos_json": '{"a": 1}', "esquema_json": esq}))["valido"]
    assert not json.loads(validar_esquema.invoke({"datos_json": '{"a": true}', "esquema_json": esq}))["valido"]
    assert not json.loads(validar_esquema.invoke({"datos_json": "{}", "esquema_json": esq}))["valido"]


# ---------- guardas del supervisor ----------
def test_analista_sin_datos_se_redirige_al_investigador():
    sig, _, nota = aplicar_guardas(Decision(next_agent="analista", instruction="x", reason="r"), {"step_count": 0})
    assert sig == "investigador" and nota


def test_tope_de_pasos_fuerza_finish():
    sig, _, _ = aplicar_guardas(Decision(next_agent="investigador", instruction="x", reason="r"), {"step_count": MAX_STEPS})
    assert sig == "FINISH"


def test_revisiones_agotadas_cierra():
    estado = {
        "step_count": 3, "research_notes": "n", "revisions": {"investigador": 1},
        "contributions": [{"agent": "investigador"}, {"agent": "analista"}],
    }  # fmt: skip
    sig, _, _ = aplicar_guardas(Decision(next_agent="investigador", instruction="x", reason="r"), estado)
    assert sig == "FINISH"


# ---------- rúbrica ----------
def test_rubrica_detecta_faltantes():
    assert len(revisar_rubrica({})) == 2
    ok = {
        "research_notes": "dato [doc-001]", "analysis_result": "media 3.4",
        "contributions": [{"agent": "analista", "tools_used": ["calcular_estadisticas"]}],
    }  # fmt: skip
    assert revisar_rubrica(ok) == []
    sin_tools = {**ok, "contributions": [{"agent": "analista", "tools_used": []}]}
    assert any("herramientas" in p for p in revisar_rubrica(sin_tools))


def test_validacion_no_reintenta_indefinidamente():
    base = {"validation_feedback": "falla", "step_count": 1}
    assert route_after_validation({**base, "validation_attempts": 1}) == "supervisor"
    assert route_after_validation({**base, "validation_attempts": 2}) == "sintetizador"


# ---------- flujo completo ----------
def test_flujo_delegacion_investigador_analista_finish():
    sup, inv, ana = modelos_demo()
    final = ejecutar(build_graph(sup, inv, ana), "pregunta", verbose=False)
    assert [c["agent"] for c in final["contributions"]] == ["investigador", "analista"]
    assert final["task_completed"] and "3.4" in final["final_answer"]
    assert final["validation_feedback"] == ""
    assert final["step_count"] == 3


def test_supervisor_infinito_termina_por_tope_de_pasos():
    """Un supervisor que nunca quiere terminar no puede colgar el grafo."""
    sup = ScriptedChat(
        decisions=[Decision(next_agent="investigador", instruction="busca más", reason="siempre más")],
        replies=[AIMessage(content="respuesta parcial")],
    )
    inv = ScriptedChat(replies=[tool_call("buscar_en_base_conocimiento", {"consulta": "aurorabook"}, "c"), AIMessage(content="[doc-001] algo")])
    ana = ScriptedChat(replies=[AIMessage(content="sin datos")])
    final = ejecutar(build_graph(sup, inv, ana), "pregunta", verbose=False)
    assert final["task_completed"]
    assert final["step_count"] <= MAX_STEPS
    assert "analista" in final["validation_feedback"]  # la rúbrica dejó constancia en la respuesta final


def test_validacion_rechaza_y_vuelve_al_supervisor():
    sup = ScriptedChat(
        decisions=[
            Decision(next_agent="investigador", instruction="busca", reason="r"),
            Decision(next_agent="FINISH", reason="cierro sin analizar"),  # error del LLM
            Decision(next_agent="analista", instruction="analiza", reason="feedback de validación"),
            Decision(next_agent="FINISH", reason="ahora sí"),
        ],
        replies=[AIMessage(content="final")],
    )
    inv = ScriptedChat(replies=[AIMessage(content="datos [doc-001]")])
    ana = ScriptedChat(replies=[tool_call("calcular_estadisticas", {"valores": [1, 2, 3]}, "c"), AIMessage(content="media 2.0")])
    final = ejecutar(build_graph(sup, inv, ana), "pregunta", verbose=False)
    assert [c["agent"] for c in final["contributions"]] == ["investigador", "analista"]
    assert final["validation_feedback"] == "" and final["validation_attempts"] == 2


def test_especialista_recibe_contexto_aislado():
    """El analista ve la instrucción + notas del investigador, no la pregunta cruda ni los mensajes del supervisor."""
    from langchain_core.messages import HumanMessage

    vistos = []

    class Espia(ScriptedChat):
        def _generate(self, messages, *a, **k):
            vistos.append(messages)
            return super()._generate(messages, *a, **k)

    sup, inv, _ = modelos_demo()
    ana = Espia(replies=[tool_call("calcular_estadisticas", {"valores": [1, 2]}, "c"), AIMessage(content="media 1.5")])
    ejecutar(build_graph(sup, inv, ana), "PREGUNTA_ORIGINAL_SECRETA", verbose=False)
    humanos = [m.content for m in vistos[0] if isinstance(m, HumanMessage)]
    assert len(humanos) == 1 and "PREGUNTA_ORIGINAL_SECRETA" not in humanos[0] and "[doc-001]" in humanos[0]
    assert not any(getattr(m, "name", None) == "supervisor" for m in vistos[0])

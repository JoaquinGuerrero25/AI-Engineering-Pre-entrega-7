"""Tests de la API: redis falso + MemorySaver + LLMs guionados (sin red, sin API keys, sin Redis real)."""

import time

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.hitl import clasificar_criticidad
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("PHOENIX_DISABLED", "1")
    monkeypatch.setenv("WORKER_CONCURRENCY", "3")
    app.state.redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    app.state.checkpointer = MemorySaver()
    with TestClient(app) as c:
        yield c
    del app.state.redis, app.state.checkpointer


def esperar(client, job_id, estados, timeout=20):
    fin = time.time() + timeout
    while time.time() < fin:
        job = client.get(f"/tasks/{job_id}").json()
        if job["status"] in estados:
            return job
        time.sleep(0.1)
    raise AssertionError(f"timeout; último estado: {job['status']}")


def test_clasificador_de_criticidad():
    assert clasificar_criticidad("Investiga las reseñas y calcula el promedio") == ""
    assert "efectos secundarios" in clasificar_criticidad("Envía un correo con el reporte a todos los clientes")
    assert "efectos secundarios" in clasificar_criticidad("Emití un reembolso")
    assert clasificar_criticidad("hola", forzada=True)


def test_post_encola_y_devuelve_job_id_enseguida(client):
    r = client.post("/tasks", json={"query": "Reseñas de la AuroraBook X14, sentimiento y promedio"})
    assert r.status_code == 202
    assert r.json()["status"] == "PENDING" and r.json()["job_id"]
    assert client.get("/tasks/inexistente").status_code == 404


def test_flujo_normal_termina_en_done(client):
    jid = client.post("/tasks", json={"query": "Reseñas de la AuroraBook X14, sentimiento y promedio"}).json()["job_id"]
    job = esperar(client, jid, {"DONE", "FAILED", "WAITING_APPROVAL"})
    assert job["status"] == "DONE" and "AuroraBook" in job["result"]
    assert job["action_log"] == []


def test_tarea_critica_pausa_hasta_la_aprobacion(client):
    jid = client.post("/tasks", json={"query": "Reseñas de la AuroraBook X14 y envía el resumen por correo"}).json()["job_id"]
    job = esperar(client, jid, {"WAITING_APPROVAL", "DONE", "FAILED"})
    assert job["status"] == "WAITING_APPROVAL"
    assert job["approval_request"]["tipo"] == "aprobacion_requerida"
    time.sleep(0.5)  # sigue pausado: nadie aprobó
    assert client.get(f"/tasks/{jid}").json()["status"] == "WAITING_APPROVAL"

    r = client.post(f"/tasks/{jid}/approve", json={"approved": True, "reviewer": "ana"})
    assert r.status_code == 202
    job = esperar(client, jid, {"DONE", "FAILED", "REJECTED"})
    assert job["status"] == "DONE"
    assert job["action_log"] and "ana" in job["action_log"][0]
    assert client.post(f"/tasks/{jid}/approve", json={"approved": True}).status_code == 409  # ya resuelto


def test_rechazo_no_ejecuta_la_accion(client):
    jid = client.post("/tasks", json={"query": "Reseñas de la AuroraBook X14", "require_approval": True}).json()["job_id"]
    esperar(client, jid, {"WAITING_APPROVAL"})
    client.post(f"/tasks/{jid}/approve", json={"approved": False, "reviewer": "ana", "comment": "no"})
    job = esperar(client, jid, {"REJECTED", "DONE", "FAILED"})
    assert job["status"] == "REJECTED" and job["action_log"] == []


def test_aprobar_job_no_pausado_da_409(client):
    jid = client.post("/tasks", json={"query": "Reseñas de la AuroraBook X14, sentimiento y promedio"}).json()["job_id"]
    esperar(client, jid, {"DONE"})
    assert client.post(f"/tasks/{jid}/approve", json={"approved": True}).status_code == 409
    assert client.post("/tasks/nope/approve", json={"approved": True}).status_code == 404


def test_excepcion_en_el_agente_deja_el_job_en_failed(client, monkeypatch):
    import app.main as m

    def explota(_cp):
        raise RuntimeError("el LLM se cayó")

    monkeypatch.setattr(m, "crear_grafo", explota)
    # los workers ya capturaron `crear_grafo` por la lambda de lifespan -> reiniciamos el cliente
    app.state.redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    with TestClient(app) as c2:
        jid = c2.post("/tasks", json={"query": "cualquier tarea"}).json()["job_id"]
        job = esperar(c2, jid, {"FAILED", "DONE"})
    assert job["status"] == "FAILED" and "el LLM se cayó" in job["error"]

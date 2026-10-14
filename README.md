# Pre-entrega 7 · API de producción y monitoreo activo

API REST (**FastAPI**) que expone el orquestador multi-agente del Módulo 6 (Supervisor + Investigador +
Analista) con:

| Capacidad | Cómo |
|---|---|
| Endpoints asíncronos | `POST /tasks` encola en Redis y devuelve un `job_id` al instante; `GET /tasks/{id}` consulta el estado. Redis con cliente `redis.asyncio` y `grafo.ainvoke`: nada bloquea el event loop. |
| Estado en Redis | Estado de cada job (`PENDING/RUNNING/WAITING_APPROVAL/DONE/REJECTED/FAILED`), cola de trabajo y **checkpoints de LangGraph** (`AsyncRedisSaver`, `thread_id = job_id`). |
| Observabilidad | **Arize Phoenix** vía OpenInference: traza por job con un span por nodo, llamada al LLM (tokens, modelo → costo) y herramienta. |
| Human-in-the-loop | Nodo `aprobacion_humana` con `interrupt()`: las tareas críticas se pausan hasta `POST /tasks/{id}/approve`. |
| Errores | Cualquier excepción del agente deja el job en `FAILED` con el error en Redis (el polling siempre termina). |

## Arquitectura

```
cliente ──POST /tasks──▶ FastAPI ──RPUSH──▶ Redis (cola)
   ▲                        │                   │ BLPOP
   │ GET /tasks/{id}        ▼                   ▼
   └──────────────── Redis (estado) ◀── workers asyncio (WORKER_CONCURRENCY=5)
                                                │ ainvoke(thread_id=job_id)
                                                ▼
   Supervisor ⇄ Investigador / Analista → Validación → ◇ aprobación humana → [acción] → Síntesis
        │ checkpoints en Redis (RedisSaver)        │ trazas OTLP → Phoenix (:6006)
```

El grafo es el del M6 más una compuerta HITL entre la validación y la síntesis:

```
validacion ──▶ aprobacion_humana ──(no crítica)──────────────▶ sintetizador ──▶ END
                      │ (crítica: interrupt → WAITING_APPROVAL)
                      ├─ aprobada ──▶ ejecutar_accion ───────▶ sintetizador
                      └─ rechazada ─▶ END (status REJECTED, sin efectos)
```

Una tarea es **crítica** si pide una acción con efectos secundarios (enviar, emitir, reembolsar,
eliminar, cancelar, pagar, publicar… ver `ACCIONES_CRITICAS` en [`app/hitl.py`](app/hitl.py)) o si el
cliente manda `"require_approval": true`. Al pausar, el estado queda en el checkpoint de Redis; el
endpoint de aprobación reanuda con `Command(resume=...)`, incluso si el proceso se reinició.

## Estructura

```
app/
├── main.py            # FastAPI: POST /tasks, GET /tasks/{id}, /health + arranque de workers
├── graph.py           # orquestador M6 + AsyncRedisSaver + compuerta HITL
├── worker.py          # consume la cola, corre el grafo, actualiza el estado (FAILED ante excepción)
├── store.py           # JobStore: estado y cola en Redis (async)
├── observability.py   # init de Phoenix (OpenInference / LangChain instrumentor)
├── hitl.py            # nodo de aprobación (interrupt) + POST /tasks/{id}/approve
├── state.py, supervisor.py, validation.py, llm.py, fakes.py, agents/, data/   # sistema del M6
scripts/load_test.py   # 5 peticiones concurrentes + latencias/p95
tests/                 # 21 tests sin red (fakeredis + LLM guionado)
screenshots/           # capturas de Phoenix (trazas, costo, p95)
```

## Cómo levantarlo

Requisitos: Python 3.12+, Docker. **Redis debe ser Redis Stack** (RedisJSON + RediSearch), que es lo que
necesita `RedisSaver`; el `docker-compose.yml` ya usa esa imagen.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # completá UNA clave de LLM (o LLM_PROVIDER=mock para probar sin claves)
docker compose up -d redis phoenix
uvicorn app.main:app --port 8000
```

- API y docs interactivas: http://localhost:8000/docs
- Phoenix (trazas, costos, latencias): http://localhost:6006

Todo en contenedores (API incluida): `docker compose up --build` (usa el `.env` para la clave del LLM).

### Probar a mano

```bash
curl -X POST localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"query": "Investiga las reseñas de la AuroraBook X14 y calcula la calificación promedio"}'
# -> {"job_id": "…", "status": "PENDING"}
curl localhost:8000/tasks/<job_id>        # PENDING → RUNNING → DONE (o FAILED con "error")
```

Tarea crítica (HITL):

```bash
curl -X POST localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"query": "Investiga las reseñas de la AuroraBook X14 y envía un resumen por correo al equipo"}'
curl localhost:8000/tasks/<job_id>        # status: WAITING_APPROVAL + approval_request
curl -X POST localhost:8000/tasks/<job_id>/approve -H "Content-Type: application/json" \
  -d '{"approved": true, "reviewer": "joaquin"}'     # approved:false -> status REJECTED
```

## Prueba de carga: 5 peticiones concurrentes

Con la API y Phoenix levantados:

```bash
python scripts/load_test.py -n 5 --auto-approve
```

Lanza 5 `POST /tasks` a la vez (una de ellas crítica, que el script aprueba sola), espera a que todas
terminen e imprime el estado, la latencia de encolado y la latencia extremo a extremo (p50 / p95 / máx).

Luego, en Phoenix (http://localhost:6006 → proyecto `agente-multiagente-prod`):

1. **Traces**: una traza por job; al abrirla se ven los nodos (`supervisor`, `investigador`, `analista`, `validacion`, `aprobacion_humana`, `sintetizador`) y cada llamada al LLM.
2. **Costo por ejecución**: columna *Cost* de la tabla de trazas / pestaña *Metrics* (Phoenix lo calcula con los tokens de entrada y salida y el modelo).
3. **Latencia p95**: panel *Latency* (P50/P99 y percentiles) del proyecto.
4. Nodo que más latencia/tokens consume: ordená los spans de una traza por *Latency* y *Total tokens*.

Las capturas van en [`screenshots/`](screenshots/).

## Tests

```bash
pytest -q
```

Cubren herramientas, guardas del supervisor, flujos del grafo (M6) y la API: encolado inmediato,
`DONE`, pausa HITL → aprobación → acción, rechazo, doble aprobación (409) y `FAILED` ante excepción.
Usan `fakeredis`, `MemorySaver` y LLMs guionados (`LLM_PROVIDER=mock`), así que no requieren red.

## Decisiones de diseño

- **Cola en Redis + workers asyncio** (en lugar de `BackgroundTasks`): la cola sobrevive al proceso de la API y la concurrencia se acota con `WORKER_CONCURRENCY`. Para escalar horizontalmente basta con más instancias de la API (o mover `worker.py` a un proceso aparte / Arq).
- **Nodos síncronos del M6 sin reescribir**: `ainvoke` los ejecuta en el threadpool, por lo que las llamadas al LLM no bloquean el event loop.
- **Idempotencia del nodo HITL**: al reanudar, el nodo se re-ejecuta desde el inicio; por eso antes de `interrupt()` no hay efectos secundarios (la acción real vive en `ejecutar_accion`, posterior a la aprobación).
- **`MAX_STEPS`, guardas y rúbrica** del M6 se conservan: el costo por ejecución queda acotado.

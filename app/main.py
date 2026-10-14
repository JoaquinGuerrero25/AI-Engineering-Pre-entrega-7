"""API FastAPI: POST /tasks encola y responde al instante; GET /tasks/{id} consulta el estado.

Uso:  uvicorn app.main:app --port 8000
"""

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis.asyncio import Redis

load_dotenv()

from app import hitl  # noqa: E402
from app.graph import crear_grafo  # noqa: E402
from app.observability import init_observability  # noqa: E402
from app.store import JobStore  # noqa: E402
from app.worker import worker_loop  # noqa: E402


class TaskIn(BaseModel):
    query: str = Field(min_length=3, examples=["Investiga las reseñas de la AuroraBook X14 y calcula la calificación promedio."])
    require_approval: bool = Field(False, description="Fuerza la pausa HITL aunque la tarea no se detecte como crítica.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    concurrency = int(os.getenv("WORKER_CONCURRENCY", "5"))
    init_observability()
    redis: Redis = getattr(app.state, "redis", None) or Redis.from_url(redis_url, decode_responses=True)
    app.state.jobs = JobStore(redis)

    # Checkpointer de Redis (requiere Redis Stack: RedisJSON + RediSearch). Los tests inyectan uno propio.
    checkpointer = getattr(app.state, "checkpointer", None)
    cm = None
    if checkpointer is None:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        cm = AsyncRedisSaver.from_conn_string(redis_url)
        checkpointer = await cm.__aenter__()
        await checkpointer.asetup()

    workers = [
        asyncio.create_task(worker_loop(f"w{i}", app.state.jobs, lambda: crear_grafo(checkpointer)))
        for i in range(concurrency)
    ]
    try:
        yield
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if cm:
            await cm.__aexit__(None, None, None)
        await redis.aclose()


app = FastAPI(title="API multi-agente (producción)", version="1.0.0", lifespan=lifespan)
app.include_router(hitl.router)


@app.post("/tasks", status_code=202)
async def crear_tarea(body: TaskIn) -> dict:
    """Encola la tarea y devuelve el job_id sin esperar al agente."""
    jobs: JobStore = app.state.jobs
    job_id = await jobs.create(body.query, body.require_approval)
    await jobs.enqueue({"job_id": job_id})
    return {"job_id": job_id, "status": "PENDING"}


@app.get("/tasks/{job_id}")
async def estado_tarea(job_id: str) -> dict:
    job = await app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job inexistente")
    return job


@app.get("/health")
async def health() -> dict:
    await app.state.jobs.r.ping()
    return {"status": "ok"}

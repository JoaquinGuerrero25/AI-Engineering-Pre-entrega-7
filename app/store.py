"""Estado de los jobs y cola de trabajo en Redis (cliente asíncrono: nunca bloquea el event loop)."""

import json
import time
import uuid
from typing import Any

from redis.asyncio import Redis

QUEUE_KEY = "agent:queue"
JOB_TTL_S = 60 * 60 * 24  # los jobs expiran a las 24 h

PENDING = "PENDING"
RUNNING = "RUNNING"
WAITING_APPROVAL = "WAITING_APPROVAL"
DONE = "DONE"
FAILED = "FAILED"
REJECTED = "REJECTED"

_TEXTO = ("status", "query", "job_id", "error", "result")  # campos guardados como texto plano


def _key(job_id: str) -> str:
    return f"agent:job:{job_id}"


class JobStore:
    def __init__(self, redis: Redis):
        self.r = redis

    async def create(self, query: str, force_approval: bool = False) -> str:
        job_id = uuid.uuid4().hex
        now = str(time.time())
        await self.r.hset(
            _key(job_id),
            mapping={
                "job_id": job_id,
                "status": PENDING,
                "query": query,
                "force_approval": json.dumps(force_approval),
                "created_at": now,
                "updated_at": now,
            },
        )
        await self.r.expire(_key(job_id), JOB_TTL_S)
        return job_id

    async def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        mapping = {k: v if k in _TEXTO else json.dumps(v, default=str) for k, v in fields.items()}
        await self.r.hset(_key(job_id), mapping=mapping)

    async def get(self, job_id: str) -> dict | None:
        raw = await self.r.hgetall(_key(job_id))
        if not raw:
            return None
        return {k: v if k in _TEXTO else json.loads(v) for k, v in raw.items()}

    async def claim_approval(self, job_id: str) -> bool:
        """SET NX atómico: solo una aprobación concurrente gana."""
        return bool(await self.r.set(f"agent:approve-lock:{job_id}", "1", nx=True, ex=JOB_TTL_S))

    async def enqueue(self, item: dict) -> None:
        await self.r.rpush(QUEUE_KEY, json.dumps(item))

    async def dequeue(self, timeout: int = 1) -> dict | None:
        res = await self.r.blpop([QUEUE_KEY], timeout=timeout)
        return json.loads(res[1]) if res else None

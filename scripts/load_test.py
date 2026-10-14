"""Prueba de carga: N peticiones concurrentes contra la API (por defecto 5).

Lanza todas a la vez, espera a que terminen (aprobando automáticamente las que queden en
WAITING_APPROVAL si se pasa --auto-approve) e imprime latencias y p95.

    python scripts/load_test.py                       # 5 peticiones
    python scripts/load_test.py -n 5 --auto-approve   # incluye una tarea crítica
"""

import argparse
import asyncio
import math
import time

import httpx

CONSULTAS = [
    "Investiga las reseñas de la notebook AuroraBook X14, analiza el sentimiento de los clientes y calcula la calificación promedio.",
    "Busca las ventas de la AuroraBook X14 y calcula el ingreso total.",
    "Investiga la política de devoluciones y resume los plazos con cifras.",
    "Analiza las reseñas de la AuroraBook X14 y calcula la desviación estándar de las calificaciones.",
    "Investiga las reseñas de la AuroraBook X14 y envía un resumen por correo al equipo de producto.",  # crítica -> HITL
]
TERMINALES = {"DONE", "FAILED", "REJECTED"}


def percentil(valores: list[float], p: float) -> float:
    orden = sorted(valores)
    return orden[max(0, math.ceil(p / 100 * len(orden)) - 1)]


async def una_peticion(cli: httpx.AsyncClient, i: int, consulta: str, auto_approve: bool) -> dict:
    t0 = time.perf_counter()
    r = await cli.post("/tasks", json={"query": consulta})
    r.raise_for_status()
    post_ms = (time.perf_counter() - t0) * 1000
    job_id = r.json()["job_id"]
    aprobado = False
    while True:
        job = (await cli.get(f"/tasks/{job_id}")).json()
        if job["status"] == "WAITING_APPROVAL" and auto_approve and not aprobado:
            await cli.post(f"/tasks/{job_id}/approve", json={"approved": True, "reviewer": "load-test"})
            aprobado = True
        if job["status"] in TERMINALES:
            break
        await asyncio.sleep(0.5)
    total = time.perf_counter() - t0
    print(f"[{i}] {job['status']:<8} POST={post_ms:6.0f} ms  total={total:6.1f} s  job={job_id[:8]}")
    return {"status": job["status"], "total": total, "post_ms": post_ms}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--auto-approve", action="store_true")
    a = ap.parse_args()
    consultas = [CONSULTAS[i % len(CONSULTAS)] for i in range(a.n)]
    async with httpx.AsyncClient(base_url=a.url, timeout=30) as cli:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(una_peticion(cli, i, q, a.auto_approve) for i, q in enumerate(consultas, 1)))
        pared = time.perf_counter() - t0
    tot = [r["total"] for r in res]
    print(f"\n{a.n} peticiones concurrentes en {pared:.1f} s | " + ", ".join(f"{s}={sum(r['status']==s for r in res)}" for s in TERMINALES))
    print(f"POST /tasks (encolar) máx: {max(r['post_ms'] for r in res):.0f} ms")
    print(f"Latencia extremo a extremo: p50={percentil(tot,50):.1f}s  p95={percentil(tot,95):.1f}s  máx={max(tot):.1f}s")
    print("Costo y p95 por traza: http://localhost:6006 (proyecto agente-multiagente-prod)")


if __name__ == "__main__":
    asyncio.run(main())

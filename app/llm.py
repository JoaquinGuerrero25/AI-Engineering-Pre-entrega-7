"""Fábrica del LLM. Las claves se leen de variables de entorno (.env), nunca del código."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

MODELOS_POR_DEFECTO = {
    "openai": "gpt-4o-mini",
    "google": "gemini-3.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}


def crear_llm() -> BaseChatModel:
    """Crea el chat model según LLM_PROVIDER (openai | google | anthropic)."""
    load_dotenv()
    proveedor = os.getenv("LLM_PROVIDER", "google").lower()
    if proveedor not in MODELOS_POR_DEFECTO:
        raise ValueError(f"LLM_PROVIDER inválido: {proveedor!r}. Usá {list(MODELOS_POR_DEFECTO)}.")
    modelo = os.getenv("LLM_MODEL") or MODELOS_POR_DEFECTO[proveedor]

    # Opcional: limita el ritmo de llamadas (útil en planes gratuitos, p. ej. 5 req/min en Gemini).
    rpm = os.getenv("LLM_REQUESTS_PER_MINUTE")
    limitador = InMemoryRateLimiter(requests_per_second=float(rpm) / 60, max_bucket_size=1) if rpm else None

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=modelo, temperature=0, rate_limiter=limitador)
    if proveedor == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=modelo, temperature=0, max_retries=6, rate_limiter=limitador)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=modelo, temperature=0, rate_limiter=limitador)

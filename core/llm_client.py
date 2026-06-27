from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_DOUBAO_MODEL = "doubao-1-5-pro-32k-250115"


@dataclass(frozen=True)
class DoubaoConfig:
    api_key: str
    base_url: str = DEFAULT_ARK_BASE_URL
    model: str = DEFAULT_DOUBAO_MODEL


def load_doubao_config() -> DoubaoConfig:
    load_dotenv()

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 ARK_API_KEY。请先在环境变量或 .env 文件中配置豆包 API Key。"
        )

    return DoubaoConfig(
        api_key=api_key,
        base_url=os.environ.get("ARK_BASE_URL", DEFAULT_ARK_BASE_URL),
        model=os.environ.get("ARK_MODEL", DEFAULT_DOUBAO_MODEL),
    )


def create_doubao_client(config: DoubaoConfig | None = None) -> OpenAI:
    config = config or load_doubao_config()
    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )


def call_doubao_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    client: OpenAI | None = None,
) -> str:
    config = load_doubao_config()
    client = client or create_doubao_client(config)

    completion = client.chat.completions.create(
        model=model or config.model,
        messages=messages,
        temperature=temperature,
    )

    content = completion.choices[0].message.content
    return content or ""


def call_doubao_chat_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    client: OpenAI | None = None,
) -> str:
    config = load_doubao_config()
    client = client or create_doubao_client(config)

    completion = client.chat.completions.create(
        model=model or config.model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    return content or ""


def build_basic_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

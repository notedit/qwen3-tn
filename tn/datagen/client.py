"""DeepSeek API 异步客户端:thinking 开关、指数退避重试、并发限流。"""

import asyncio
import json
import os
import re

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def load_dotenv(path: str = ".env") -> None:
    """launch.sh 已注入 env;直接跑脚本时兜底读 .env。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class DeepSeekClient:
    def __init__(self, concurrency: int = 8):
        load_dotenv()
        self.client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=180,
            max_retries=0,  # 重试自己控制
        )
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.sem = asyncio.Semaphore(concurrency)
        self.n_requests = 0
        self.n_retried_requests = 0

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    async def _chat_once(self, **kwargs):
        async with self.sem:
            self.n_requests += 1
            return await self.client.chat.completions.create(**kwargs)

    async def chat(self, messages, *, thinking=False, temperature=0.8,
                   max_tokens=2000, json_mode=True) -> str:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._chat_once(**kwargs)
        return resp.choices[0].message.content or ""

    async def chat_json(self, messages, **kw) -> dict | None:
        text = await self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
            return None

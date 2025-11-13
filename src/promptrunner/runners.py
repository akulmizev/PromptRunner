import asyncio
import json
import os
from abc import ABC, abstractmethod
from tqdm.asyncio import tqdm as atqdm
from typing import Any, Dict, List

from .adapters import TogetherAIAdapter, LiteLLMAdapter, OllamaAdapter


class BaseRunner(ABC):

    def __init__(
        self,
        model: str,
        backend: str,
        api_key: str = None,
        response_kwargs: Dict[str, Any] = None
    ):
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self.response_kwargs = response_kwargs or dict()

        adapter_map = {
            "togetherai": TogetherAIAdapter,
            "litellm": LiteLLMAdapter,
            "ollama": OllamaAdapter,
        }

        adapter_class = adapter_map.get(backend)
        if adapter_class is None:
            raise ValueError(f"Unknown backend '{backend}'. Valid options: {', '.join(adapter_map.keys())}")

        if backend in ["togetherai", "litellm"] and not api_key:
            raise ValueError(f"API key required for '{backend}' backend.")

        self.adapter = adapter_class(model, api_key) if api_key else adapter_class(model)

    @abstractmethod
    async def run(self, message_list: List[Dict[str, Any]], **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")


class AsyncRunner(BaseRunner):

    async def run_async(self, message_list: List[Dict[str, Any]], max_concurrent: int = 1, **kwargs) -> List[Any]:
        """Async implementation of run."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(messages):
            async with semaphore:
                query = {"messages": messages, **self.response_kwargs}
                return await self.adapter.chat_completion(query)

        tasks = [run_with_semaphore(messages) for messages in message_list]
        return await atqdm.gather(*tasks, desc="Running prompts...")

    def run(self, message_list: List[Dict[str, Any]], max_concurrent: int = 1, **kwargs) -> List[Any]:
        """Synchronous wrapper that can be called without asyncio.run()."""
        return asyncio.run(self.run_async(message_list, max_concurrent, **kwargs))

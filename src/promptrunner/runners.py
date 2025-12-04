import asyncio
import json
import tempfile
import time
from abc import ABC, abstractmethod
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm as stqdm
from typing import Any, Dict, List, Type

from .adapters import LLMClientAdapter, TogetherAIAdapter, LiteLLMAdapter, OllamaAdapter


class BaseRunner(ABC):

    ADAPTER_MAP: Dict[str, Type[OllamaAdapter]] = {
        "ollama": OllamaAdapter,
        "togetherai": TogetherAIAdapter,
        "litellm": LiteLLMAdapter,
    }

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

    def _initialize_adapter(self, async_mode: bool = False) -> LLMClientAdapter:

        adapter_class = self.ADAPTER_MAP.get(self.backend)

        if adapter_class is None:
            raise ValueError(f"Unknown backend '{self.backend}'. Valid options: {', '.join(self.ADAPTER_MAP.keys())}")

        if self.backend in ["togetherai", "litellm"] and not self.api_key:
            raise ValueError(f"API key required for '{self.backend}' backend.")

        init_kwargs = {
            "model": self.model,
            "async_mode": async_mode
        }

        if self.api_key:
            init_kwargs["api_key"] = self.api_key

        return adapter_class(**init_kwargs)

    @abstractmethod
    def run(self, message_list: List[Dict[str, Any]], **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")


class AsyncRunner(BaseRunner):
    """
    Handles concurrent asynchronous execution of queries using the adapter's
    async API (chat_completion_async).
    """

    def __init__(
            self,
            model: str,
            backend: str,
            api_key: str = None,
            response_kwargs: Dict[str, Any] = None
    ) -> None:
        super().__init__(model, backend, api_key, response_kwargs)
        self.adapter = self._initialize_adapter(async_mode=True)

    async def run_async(self, message_list: List[Dict[str, Any]], max_concurrent: int = 1, **kwargs) -> List[Any]:
        """Core async implementation."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(messages):
            async with semaphore:
                query = {"messages": messages, **self.response_kwargs}
                # Use the adapter's async method
                return await self.adapter.chat_completion_async(query, **kwargs)

        tasks = [run_with_semaphore(messages) for messages in message_list]
        return await atqdm.gather(*tasks, desc="Running prompts (async)...")

    def run(self, message_list: List[Dict[str, Any]], max_concurrent: int = 1, **kwargs) -> List[Any]:
        """Synchronous wrapper for easy execution."""
        return asyncio.run(self.run_async(message_list, max_concurrent, **kwargs))


class SyncRunner(BaseRunner):
    """
    Handles sequential synchronous execution of queries using the adapter's
    sync API (chat_completion).
    """

    def __init__(
            self,
            model: str,
            backend: str,
            api_key: str = None,
            response_kwargs: Dict[str, Any] = None
    ) -> None:
        super().__init__(model, backend, api_key, response_kwargs)

        # Use the utility function to initialize the adapter in SYNC mode
        self.adapter = self._initialize_adapter(async_mode=False)

    def run(self, message_list: List[Dict[str, Any]], **kwargs) -> List[Any]:
        """
        Processes a list of message queries sequentially using the adapter's
        synchronous method.
        """
        results = []

        for messages in stqdm(message_list, desc="Running prompts (sync)..."):
            query = {"messages": messages, **self.response_kwargs}
            result = self.adapter.chat_completion(query, **kwargs)
            results.append(result)

        return results


class BatchRunner(BaseRunner):

    def __init__(
            self,
            model: str,
            backend: str,
            api_key: str = None,
            response_kwargs: Dict[str, Any] = None
    ) -> None:

        if backend != "togetherai":
            raise ValueError("Only togetherai backend is supported for batching.")

        super().__init__(model, backend, api_key, response_kwargs)
        self.adapter = self._initialize_adapter(async_mode=False)

    def run(
        self,
        message_list: List[Dict[str, Any]],
        request_ids: List[str] = None,
        poll_interval: int = 30,
        output_path: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:

        if request_ids is None:
            request_ids = [f"batch_id_{i+1}" for i in range(len(message_list))]
            print(f"Request IDs not provided, using arbitrary IDs instead.")

        request_list = [
            {"custom_id": request_id, "body": {"model": self.model, "messages": messages}} \
            for request_id, messages in zip(request_ids, message_list)
        ]

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            batch_filename = f.name
            for request in request_list:
                f.write(json.dumps(request) + "\n")

        batch_id = self.adapter.submit_batch(batch_file=batch_filename)

        print(f"\nSuccessfully submitted batch job {batch_id}...")

        print("\nWaiting for batch job to complete (polling)...")
        get_status = getattr(self.adapter, 'check_batch_status')

        while True:
            status = get_status(batch_id)
            print(f"   -> Current Status: {status}")

            if status == "COMPLETED":
                print("\nBatch job completed.")
                break
            elif status in ["FAILED", "CANCELLED"]:
                raise RuntimeError(f"Batch job failed or was cancelled. Status: {status}")

            time.sleep(poll_interval)

        file_context = open(output_path, "w+") if output_path else tempfile.NamedTemporaryFile(mode='w+', delete=False)
        with file_context as f:
            output_filename = f.name
            responses = self.adapter.retrieve_batch(batch_id=batch_id, output_file=output_filename)

        return responses

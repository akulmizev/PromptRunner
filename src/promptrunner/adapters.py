import json
import regex as re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

from litellm import acompletion, completion
from ollama import AsyncClient, Client
from openai import OpenAI, AsyncOpenAI
from together import AsyncTogether, Together
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.messages.batch_create_params import Request as AnthropicBatchRequest


THINK_PATTERN = r'<think>(.*?)</think>'

PROVIDER_CONFIG = {
    "ollama": {
        "protocol": "ollama",
        "requires_api_key": False,
        "supports_batch": False,
    },
    "togetherai": {
        "protocol": "togetherai",
        "requires_api_key": True,
        "supports_batch": True,
    },
    "openai": {
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "requires_api_key": True,
        "supports_batch": True,
    },
    "anthropic": {
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com",
        "requires_api_key": True,
        "supports_batch": True,
    },
    "google": {
        "protocol": "google",
        "requires_api_key": True,
        "supports_batch": True,
    },
    "novitaai": {
        "protocol": "openai",
        "base_url": "https://api.novita.ai/v3/openai",
        "requires_api_key": True,
        "supports_batch": True,
    },
    "openrouter": {
        "protocol": "openai",
        "base_url": "https://openrouter.ai/api",
        "requires_api_key": True,
        "supports_batch": False,
    },
}


class LLMClientAdapter(ABC):

    def __init__(self, model: str, async_mode: bool = False):
        self.model = model
        self.async_mode = async_mode

    @abstractmethod
    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        pass

    @abstractmethod
    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        pass

    @abstractmethod
    def _process_response(self, response: Any) -> Dict[str, str]:
        pass

    def chat_completion(self, query: Dict[str, Any], **kwargs) -> Dict[str, str]:
        assert "messages" in query

        if self.async_mode:
            raise RuntimeError("Adapter initialized in async mode. Use chat_completion_async().")

        raw_response = self._call_client(query, **kwargs)
        return self._process_response(raw_response)

    async def chat_completion_async(self, query: Dict[str, Any], **kwargs) -> Dict[str, str]:
        assert "messages" in query

        if not self.async_mode:
            raise RuntimeError("Adapter not initialized in async mode. Use chat_completion().")

        raw_response = await self._call_client_async(query, **kwargs)
        return self._process_response(raw_response)

    @staticmethod
    def clean_content(content: str) -> str:
        return str(re.sub(THINK_PATTERN, '', content, flags=re.DOTALL).strip())

    @staticmethod
    def extract_reasoning_trace(content: str) -> Optional[str]:
        reasoning_trace = None
        think_matches = re.findall(THINK_PATTERN, content, re.DOTALL)

        if think_matches:
            reasoning_trace = '\n'.join(think_matches)

        return reasoning_trace

    def submit_batch(self, **kwargs) -> str:
        raise NotImplementedError(f"Batching not supported by {self.__class__.__name__}.")

    def check_batch_status(self, **kwargs) -> str:
        raise NotImplementedError(f"Batching not supported by {self.__class__.__name__}.")

    def retrieve_batch(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError(f"Batching not supported by {self.__class__.__name__}.")


class OllamaProtocolAdapter(LLMClientAdapter):
    """Protocol adapter for Ollama (local models)."""

    def __init__(self, model: str, async_mode: bool = False) -> None:
        super().__init__(model, async_mode)
        self.client: Union[AsyncClient, Client] = AsyncClient() if self.async_mode else Client()

    def _process_response(self, response: Any) -> Dict[str, str]:
        content = self.clean_content(response.message.content)
        reasoning_trace = self.extract_reasoning_trace(response.message.content)
        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}

    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        messages = query.pop("messages")
        return self.client.chat(model=self.model, messages=messages, options=query, **kwargs)

    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        messages = query.pop("messages")
        return await self.client.chat(model=self.model, messages=messages, options=query, **kwargs)


class TogetherAIProtocolAdapter(LLMClientAdapter):
    """Protocol adapter for TogetherAI."""

    def __init__(self, model: str, api_key: str, async_mode: bool = False) -> None:
        super().__init__(model, async_mode)
        self.client: Union[AsyncTogether, Together] = (
            AsyncTogether(api_key=api_key) if self.async_mode else Together(api_key=api_key)
        )

    def _process_response(self, response: Any) -> Dict[str, str]:
        content_raw = response.choices[0].message.content
        content = self.clean_content(content_raw)
        reasoning_trace = self.extract_reasoning_trace(content_raw)
        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}

    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        return self.client.chat.completions.create(model=self.model, **query, **kwargs)

    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        return await self.client.chat.completions.create(model=self.model, **query, **kwargs)

    def submit_batch(self, batch_file: str) -> str:
        if self.async_mode:
            raise RuntimeError("Batch operations not supported in async mode.")

        file_response = self.client.files.upload(file=batch_file, purpose="batch-api")
        batch = self.client.batches.create_batch(file_response.id, endpoint="/v1/chat/completions")
        return batch.id

    def check_batch_status(self, batch_id: str) -> str:
        batch = self.client.batches.get_batch(batch_id)
        return batch.status

    def retrieve_batch(self, batch_id: str, output_file: str) -> List[Dict[str, Any]]:
        batch = self.client.batches.get_batch(batch_id)

        if batch.status != "COMPLETED":
            raise RuntimeError("Batch job not completed.")

        self.client.files.retrieve_content(id=batch.output_file_id, output=output_file)

        output = []
        for line in open(output_file):
            response = json.loads(line)
            response_id = response["custom_id"]
            content = response["response"]["body"]["choices"][0]["message"]["content"]
            output.append({"response_id": response_id, "content": content})

        return output


class OpenAIProtocolAdapter(LLMClientAdapter):
    """Protocol adapter for OpenAI and OpenAI-compatible APIs (novitaai, openrouter, etc.)."""

    def __init__(self, model: str, api_key: str, base_url: str, async_mode: bool = False) -> None:
        super().__init__(model, async_mode)
        client_kwargs = dict(api_key=api_key, base_url=base_url)
        self.client: Union[AsyncOpenAI, OpenAI] = (
            AsyncOpenAI(**client_kwargs) if self.async_mode else OpenAI(**client_kwargs)
        )

    def _process_response(self, response: Any) -> Dict[str, str]:
        content_raw = response.choices[0].message.content
        content = self.clean_content(content_raw)
        reasoning_trace = self.extract_reasoning_trace(content_raw)
        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}

    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        return self.client.chat.completions.create(model=self.model, **query, **kwargs)

    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        return await self.client.chat.completions.create(model=self.model, **query, **kwargs)

    def submit_batch(self, batch_file: str) -> str:
        if self.async_mode:
            raise RuntimeError("Batch operations not supported in async mode.")

        sync_client = cast(OpenAI, self.client)
        file_response = sync_client.files.create(file=open(batch_file, "rb"), purpose="batch")
        batch = sync_client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch.id

    def check_batch_status(self, batch_id: str) -> str:
        sync_client = cast(OpenAI, self.client)
        batch = sync_client.batches.retrieve(batch_id)
        return batch.status.value if hasattr(batch.status, 'value') else str(batch.status)

    def retrieve_batch(self, batch_id: str, output_file: str) -> List[Dict[str, Any]]:
        sync_client = cast(OpenAI, self.client)
        batch = sync_client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch job not completed. Status: {batch.status}")

        content = sync_client.files.content(batch.output_file_id)
        output = []
        for line in content.text.strip().split('\n'):
            if line.strip():
                response = json.loads(line)
                response_id = response["custom_id"]
                content_text = response["response"]["body"]["choices"][0]["message"]["content"]
                output.append({"response_id": response_id, "content": content_text})

        return output


class AnthropicProtocolAdapter(LLMClientAdapter):
    """Protocol adapter for Anthropic."""

    def __init__(self, model: str, api_key: str, async_mode: bool = False, base_url: str = None) -> None:
        super().__init__(model, async_mode)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client: Union[AsyncAnthropic, Anthropic] = (
            AsyncAnthropic(**client_kwargs) if async_mode else Anthropic(**client_kwargs)
        )

    def _process_response(self, response: Any) -> Dict[str, str]:
        content_blocks = response.content
        content_parts = []
        reasoning_trace = None

        for block in content_blocks:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "thinking":
                if reasoning_trace is None:
                    reasoning_trace = block.thinking
                else:
                    reasoning_trace += "\n" + block.thinking

        content = self.clean_content("\n".join(content_parts))
        trace = self.extract_reasoning_trace(content) or reasoning_trace

        return {"content": str(content), "reasoning_trace": str(trace)}

    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        messages = query.pop("messages")
        params = self._build_anthropic_params(messages, query, **kwargs)
        return self.client.messages.create(**params)

    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        messages = query.pop("messages")
        params = self._build_anthropic_params(messages, query, **kwargs)
        return await self.client.messages.create(**params)

    def _build_anthropic_params(self, messages: List[Dict[str, Any]], query: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Extract system messages from the messages list (Anthropic uses top-level system parameter)
        system = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content")
            else:
                filtered_messages.append(msg)

        if system is None:
            system = query.pop("system", None)

        params = {"model": self.model, "messages": filtered_messages, **query, **kwargs}
        if system is not None:
            params["system"] = system
        if "max_tokens" not in params:
            params["max_tokens"] = 1024
        return params

    def submit_batch(self, batch_file: str) -> str:
        if self.async_mode:
            raise RuntimeError("Batch operations not supported in async mode.")

        sync_client = cast(Anthropic, self.client)

        with open(batch_file, 'r') as f:
            requests = [json.loads(line) for line in f]

        anthropic_requests = []
        for r in requests:
            body = r["body"]
            messages = body.get("messages", [])

            # Extract system messages from the messages list (Anthropic uses top-level system parameter)
            system = None
            filtered_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system = msg.get("content")
                else:
                    filtered_messages.append(msg)

            # Build new body with extracted system
            new_body = {k: v for k, v in body.items() if k != "messages"}
            new_body["messages"] = filtered_messages
            if system is not None:
                new_body["system"] = system
            if "max_tokens" not in new_body:
                new_body["max_tokens"] = 1024

            anthropic_requests.append(
                AnthropicBatchRequest(custom_id=r["custom_id"], params=new_body)
            )

        batch = sync_client.messages.batches.create(requests=anthropic_requests)
        return batch.id

    def check_batch_status(self, batch_id: str) -> str:
        sync_client = cast(Anthropic, self.client)
        batch = sync_client.messages.batches.retrieve(batch_id)
        return batch.processing_status

    def retrieve_batch(self, batch_id: str, output_file: str) -> List[Dict[str, Any]]:
        if self.async_mode:
            raise RuntimeError("Batch operations not supported in async mode.")

        sync_client = cast(Anthropic, self.client)

        batch = sync_client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            raise RuntimeError(f"Batch job not completed. Status: {batch.processing_status}")

        output = []
        for result in sync_client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                content = result.result.message.content[0].text
            else:
                content = ""
            output.append({"response_id": result.custom_id, "content": content})

        with open(output_file, 'w') as f:
            for item in output:
                f.write(json.dumps(item) + "\n")

        return output


class GoogleProtocolAdapter(LLMClientAdapter):
    """Protocol adapter for Google Gemini API (not yet implemented)."""

    def __init__(self, model: str, api_key: str, async_mode: bool = False) -> None:
        super().__init__(model, async_mode)
        raise NotImplementedError("GoogleProtocolAdapter is not yet implemented")

    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("GoogleProtocolAdapter is not yet implemented")

    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("GoogleProtocolAdapter is not yet implemented")

    def _process_response(self, response: Any) -> Dict[str, str]:
        raise NotImplementedError("GoogleProtocolAdapter is not yet implemented")


class LiteLLMProtocolAdapter(LLMClientAdapter):
    """Protocol adapter for LiteLLM (unified proxy for many providers)."""

    def __init__(self, model: str, api_key: Optional[str] = None, async_mode: bool = False):
        super().__init__(model, async_mode)
        self.api_key = api_key

    def _process_response(self, response: Any) -> Dict[str, str]:
        content = response.choices[0].message.content
        reasoning_trace = self._extract_reasoning_trace(response)
        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}

    @staticmethod
    def _extract_reasoning_trace(response: Any) -> Optional[str]:
        if hasattr(response.choices[0].message, "reasoning_trace"):
            return response.choices[0].message.reasoning_trace
        return None

    def _call_client(self, query: Dict[str, Any], **kwargs) -> Any:
        return completion(model=self.model, api_key=self.api_key, **query, **kwargs)

    async def _call_client_async(self, query: Dict[str, Any], **kwargs) -> Any:
        return await acompletion(model=self.model, api_key=self.api_key, **query, **kwargs)


# Aliases for backwards compatibility
OllamaAdapter = OllamaProtocolAdapter
LiteLLMAdapter = LiteLLMProtocolAdapter
TogetherAIAdapter = TogetherAIProtocolAdapter
NovitaAIAdapter = OpenAIProtocolAdapter
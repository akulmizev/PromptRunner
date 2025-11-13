import regex as re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from litellm import acompletion
from ollama import AsyncClient
from together import AsyncTogether



THINK_PATTERN = r'<think>(.*?)</think>'


class LLMClientAdapter(ABC):
    """Abstract Base Class for LLM client adapters."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def chat_completion(self, **kwargs) -> Dict[str, str]:
        pass

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


class TogetherAIAdapter(LLMClientAdapter):
    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        self.client = AsyncTogether(api_key=api_key)

    async def chat_completion(self, query: Dict[str, Any]) -> Dict[str, str]:
        assert "messages" in query

        response = await self.client.chat.completions.create(model=self.model, **query)
        content = self.clean_content(response.choices[0].message.content)
        reasoning_trace = self.extract_reasoning_trace(response.choices[0].message.content)

        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}


class OllamaAdapter(LLMClientAdapter):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = AsyncClient()

    async def chat_completion(self, query: Dict[str, Any]) -> Any:
        assert "messages" in query, "Ollama chat completion requires 'messages' in kwargs."
        messages = query.pop("messages")

        response = await self.client.chat(model=self.model, messages=messages, options=query)
        content = self.clean_content(response.message.content)
        reasoning_trace = self.extract_reasoning_trace(response.message.content)

        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}


class LiteLLMAdapter(LLMClientAdapter):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.api_key = api_key

    async def chat_completion(self, query: Dict[str, Any]) -> Any:
        assert "messages" in query, "LiteLLM chat completion requires 'messages' in kwargs."
        response = await acompletion(model=self.model, api_key=self.api_key, **query)
        content = response.choices[0].message.content
        reasoning_trace = self.extract_reasoning_trace(response)

        return {"content": str(content), "reasoning_trace": str(reasoning_trace)}

    def extract_reasoning_trace(self, response: Any) -> Optional[str]:
        if hasattr(response.choices[0].message, "reasoning_trace"):
            return response.choices[0].message.reasoning_trace
        else:
            return None

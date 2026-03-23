import json
import regex as re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from litellm import acompletion, completion
from ollama import AsyncClient, Client
from together import AsyncTogether, Together


THINK_PATTERN = r'<think>(.*?)</think>'


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
        raise NotImplementedError(f"Batching not supported by {self.__class__.__name__} backend.")

    def check_batch_status(self, **kwargs) -> str:
        raise NotImplementedError(f"Batching not supported by {self.__class__.__name__} backend.")

    def retrieve_batch(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError(f"Batching not supported by {self.__class__.__name__} backend.")


class TogetherAIAdapter(LLMClientAdapter):

    def __init__(self, model: str, async_mode: bool, api_key: str) -> None:

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

        self.client.files.retrieve_content(
            id=batch.output_file_id,
            output=output_file
        )

        output = []
        for line in open(output_file):
            response = json.loads(line)
            response_id = response["custom_id"]
            content = response["response"]["body"]["choices"][0]["message"]["content"]
            output.append({"response_id": response_id, "content": content})

        return output


class OllamaAdapter(LLMClientAdapter):

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


class LiteLLMAdapter(LLMClientAdapter):

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

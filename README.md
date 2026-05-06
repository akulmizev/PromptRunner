# PromptRunner

A lightweight wrapper for running inference on LLMs with a focus on data annotation tasks. Built on top of [LiteLLM](https://github.com/BerriAI/litellm), [Ollama](https://github.com/ollama/ollama-python), and provider SDKs, offering unified access to a wide variety of open- and closed-source LLMs.

## Installation

```bash
pip install git+https://github.com/akulmizev/PromptRunner
```

Or in editable mode:

```bash
pip install -e src/
```

## Protocol vs Provider

**Protocol** specifies the API client to use (e.g., OpenAI SDK, Anthropic SDK, Ollama Python client). **Provider** specifies the LLM service (e.g., novitaai, openai, anthropic, ollama).

- Most providers use the `openai` protocol (novitaai, openrouter, togetherai, etc.)
- Some have dedicated protocols: `anthropic`, `ollama`, `google`, `litellm`

The protocol is auto-detected from the provider, but can be overridden manually:

```python
runner = AsyncRunner(model="deepseek-r1", provider="novitaai")         # auto-detects openai protocol
runner = AsyncRunner(model="claude-sonnet-4", provider="anthropic")      # uses anthropic protocol
runner = AsyncRunner(model="llama3.1", provider="ollama")              # uses ollama protocol
runner = AsyncRunner(model="gpt-5", provider="openai", protocol="openai")  # explicit override
```

## API Keys

API keys are auto-loaded from environment variables. Set these in your `.bashrc`/`.zshrc`:

| Provider    | Environment Variable  |
|------------|---------------------|
| openai     | `OPENAI_API_KEY`    |
| anthropic  | `ANTHROPIC_API_KEY`|
| google     | `GOOGLE_API_KEY`   |
| togetherai | `TOGETHER_API_KEY` |
| novitaai   | `NOVITA_AI_API_KEY`|
| openrouter | `OPENROUTER_API_KEY`|
| litellm    | `LITELLM_API_KEY`  |

Or pass `api_key` explicitly to override:

```python
runner = AsyncRunner(model="gpt-4o", provider="openai", api_key="sk-...")
```

## SyncRunner

For sequential execution. Pass model parameters via `response_kwargs`:

```python
from promptrunner import SyncRunner

messages = [
    [{"role": "user", "content": "What is 2+2?"}],
]

runner = SyncRunner(
    model="llama3.1",
    provider="ollama",
    response_kwargs={"temperature": 0.7, "num_predict": 256}
)
responses = runner.run(messages)
```

## AsyncRunner

Concurrent async execution with a semaphore for rate limiting. Using OpenRouter as the main provider gives you access to 300+ models:

```python
from promptrunner import AsyncRunner

messages = [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is 3+3?"}],
]

runner = AsyncRunner(
    model="deepseek/deepseek-r1",
    provider="openrouter",
    response_kwargs={"temperature": 0.7, "max_tokens": 256}
)
responses = runner.run(messages, max_concurrent=5)

print(responses[0]["content"])
# 4
```

With reasoning traces (for models like DeepSeek R1):

```python
runner = AsyncRunner(model="deepseek/deepseek-r1", provider="openrouter")
responses = runner.run(messages)

print(responses[0]["reasoning_trace"])
# <think>
# The user is asking...
# </think>
```

## BatchRunner

For batch processing with providers that support it (openai, anthropic, togetherai, novitaai).

### Awaiting Completion

With `await_completion=True` (default), the runner blocks until all jobs complete and returns the responses:

```python
from promptrunner import BatchRunner

messages = [
    [{"role": "user", "content": f"What is {i}+{i}?"}]
    for i in range(100)
]

runner = BatchRunner(model="deepseek/deepseek-r1", provider="togetherai")
responses = runner.run(messages)

print(responses[0]["content"])
# 0
```

### Non-Blocking Mode

With `await_completion=False`, the runner submits jobs and returns immediately with batch IDs:

```python
batch_ids = runner.run(messages, await_completion=False)
print(batch_ids)
# ["batch-123", "batch-456"]
```

You can later retrieve results using `load_batch_output`:

```python
responses = runner.load_batch_output(batch_ids)
```

### Chunking

Use `max_request_load` to chunk large requests into smaller batches:

```python
runner = BatchRunner(
    model="deepseek/deepseek-r1",
    provider="togetherai",
    max_request_load=50
)

messages = [{"role": "user", "content": f"Query {i}"} for i in range(200)]
responses = runner.run(messages)
```

### Response kwargs

Pass additional parameters like temperature, max_tokens, etc. via `response_kwargs`:

```python
runner = BatchRunner(
    model="deepseek/deepseek-r1",
    provider="togetherai",
    response_kwargs={"temperature": 0.7, "max_tokens": 256}
)

messages = [{"role": "user", "content": "Tell me a story"}]
responses = runner.run(messages)
```

## Ollama Support

For local inference with Ollama, no API key required:

```python
from promptrunner import AsyncRunner

messages = [
    [{"role": "user", "content": "What is 2+2?"}],
]

runner = AsyncRunner(model="llama3.1", provider="ollama")
responses = runner.run(messages)

print(responses[0]["content"])
# 4
```

Available models depend on your local Ollama installation. Pull models with:

```bash
ollama pull llama3.1
```

## LiteLLM Protocol

Use the `litellm` protocol to access proprietary models via a LiteLLM proxy:

```python
from promptrunner import AsyncRunner

messages = [
    [{"role": "user", "content": "What is 2+2?"}],
]

runner = AsyncRunner(model="gpt-5", provider="litellm", protocol="litellm")
responses = runner.run(messages)

print(responses[0]["content"])
# 4
```

## Providers

| Provider    | Protocol   | Batch Support |
|------------|-----------|---------------|
| openai     | openai    | Yes           |
| anthropic  | anthropic | Yes           |
| google     | google    | No            |
| togetherai | togetherai| Yes           |
| novitaai   | openai    | Yes           |
| openrouter | openai    | No            |
| litellm    | litellm   | No            |
| ollama     | ollama    | No            |
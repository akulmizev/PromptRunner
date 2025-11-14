# _promptrunner_

_promptrunner_ is a tiny wrapper API built for running inference on LLMs, with a focus on data annotation tasks. It is built on top of [LiteLLM](https://github.com/BerriAI/litellm),
[Ollama](https://github.com/ollama/ollama-python), and [TogetherAI](https://github.com/togethercomputer/together-python), 
offerring direct, unified access to a wide variety of open- and closed-source LLMs. 

## Installation

To install the latest build, simply run:

``
pip install git+https://github.com/akulmizev/PromptRunner
``

## Usage

Annotate a sample of mNLI items with `gpt-5-nano`. 

```python
import os

# pip install datasets
from datasets import load_dataset

from promptrunner import AsyncRunner, NLI_BASIC_PROMPT, NLI_SYSTEM_PROMPT

# load first 10 mNLI items
mnli_sample = load_dataset("nyu-mll/multi_nli", split="train[:10]")

# format list of messages according to prompt
message_list = []
for item in mnli_sample:
    messages = [
        {
            "role": "system", "content": NLI_SYSTEM_PROMPT,
            "role": "user", "content": NLI_BASIC_PROMPT.format(item["premise"], item["hypothesis"])
        }
    ]
    message_list.append(messages)

# extract API key (assuming it is an environmental variable)
api_key = os.environ.get("OPENAI_API_KEY")

# initialize `AsyncRunner`
runner = AsyncRunner(model="gpt-5-nano", backend="litellm", api_key=api_key)

# extract responses with 3 concurrent processes
responses = runner.run(message_list, max_concurrent=3)
print(responses[0]["content"])
> "true"
```

Or use `ollama` locally instead: 

```python
# initialize `AsyncRunner`
runner = AsyncRunner(model="gpt-oss:20b", backend="ollama")

# extract responses with 3 concurrent processes
responses = runner.run(message_list, max_concurrent=3)
print(responses[0]["content"])
> "true"
```

# Buffet Bot

LLMs for AI enhanced stock research and investing for the value investor.

## Usage

```bash
python inference.py --llm anthropic
```

## Flags

- `--llm`: The LLM to use for inference. Defaults to `anthropic`. Options are `anthropic` and `openai`.

## LLM behaviours

- We've tested a lack of reproducibility in the LLMs when prompted with stock picking options. This is due to inbuilt stochasticity in the models. We can adjust this with temperature parameter but I haven't been able to get deterministic outputs yet.

## Resources
- [Design Doc (Internal)](https://docs.google.com/document/d/1ZFw9aQtlS4xDQt4nltQtCgG4GLMmYrOePZbgzkj242k/edit?usp=sharing)
- [Building a GPT-3 Enabled Research Assistant with LangChain & Pinecone](https://www.mlq.ai/gpt-3-enabled-research-assistant-langchain-pinecone/)
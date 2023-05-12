# Buffet Bot

LLMs for AI enhanced algorithmic trading.

## Usage

```bash
python main.py --config config/growth_config.py
```

## Findings
- LLMs without previous context deliver better stock picks. LLMs with no temperature sampling have less variance between simulations. See `output/experiments` for details.
- Claude with news context has lower variance than without news context. See `output/experiments/news_context` for details.
- Claude with more context has better results than Claude with less. See `output/experiments/news_context_ss_200` for details. Although it seems to have higher variance.
- Using a filtered dataset on impactful world events seems to improve performance. See `output/experiments/news_context_ss_200_filtered` for details.
- Adjusting prompt seems to have an impact on performance.

![Results](output/all_experiments_result_with_major_indices.png)

## LLM behaviours

- We've tested a lack of reproducibility in the LLMs when prompted with stock picking options. This is due to inbuilt stochasticity in the models. We can adjust this with temperature parameter.
- Sometimes Anthropic's LLM returns % allocations greater than 100%.

## Resources
- [Design Doc (Internal)](https://docs.google.com/document/d/1ZFw9aQtlS4xDQt4nltQtCgG4GLMmYrOePZbgzkj242k/edit?usp=sharing)
- [Building a GPT-3 Enabled Research Assistant with LangChain & Pinecone](https://www.mlq.ai/gpt-3-enabled-research-assistant-langchain-pinecone/)
- [Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788)
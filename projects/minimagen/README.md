# GPT from scratch

10M parameter decoder-only Transformer based character level language model. This replicates the pretraining phase of LM-based chatbots like ChatGPT and Claude.

This implementation also follows the Attention Is All You Need paper.

`input.txt` - 1M tokens of the collected works of Shakespeare
`output.txt` - LM generated text from a model trained to produce Shakespeare-sounding text.

## Resources
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
- [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
# GPT2 Shadow

Finetune a GPT2 model to produce outputs that will adversarially fool a GPT2 AI classifier. If this is achieved then we could have LLMs that are undetectable by other ML models.

## Findings

I first used a largely supervised approach using a multitask approach combining a next token prediction loss with an AI classifier loss. The model managed to game the loss function to produce non-sensical outputs whilst reducing loss and loss wasn't propagating properly through autoregressive forward functions. See `train_supervised.py` for this approach.

Then I tried an RL actor critic approach using PPO optimization algorithm to train the model where the reward signal is provided by the AI classifier. See `train_rl.py` for this approach.

Reward is going up ([W&B](https://wandb.ai/michaelliangaus/llm_rl_finetuning/runs/qn5s35jp?workspace=user-michaelliangaus))

<img src="images/reward_curve.png" width="500">

But the model is gaming the classifier. It's producing outputs that are non-sensical but are classified as human. Example of an output.

```
Hello, how are you? ( ( MTA OPEN ( Silent ( ( ( ( ( ( ( ( ( (Break ( ( ( ( ( ( ( ( monog ( ( ( Ignore ( ( ( ( ( ( ( ( ( Tr ( ( ( stoolENTSpret ( ( ( civilians']
```

To try fix this, I'm going to try add an additional loss from the RL environment that is comparing cosine similarity between the sentence embeddings of the generated output and original prompt up to that token. This should encourage the model to produce outputs that are similar in meaning to the prompt. Not sure how we handle the case that the input prompt is shorter than the generated output tho.

## Resources
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
https://huggingface.co/gpt2
https://huggingface.co/roberta-base-openai-detector
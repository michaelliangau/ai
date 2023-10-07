# GPT2

Use RL to finetune an OS LLM model to create outputs that are more similar to a target. This is a dummy project to learn more about how to RL finetune LLM models.

## Discussion
- How do we compute loss over the network?
    - Can be a terminal reward (binary signal) but how do we compute continuous loss (a measure of how good a certain generation is)?
    - Can we do a combination of some signal from goodness from AI detector (confidence value or something?) and some measure of making sure outputs are still sensical?
    - I tried naively using the ai classifier loss to train the model but the model kept hacking the loss fn to produce non-sensical outputs.

## Resources

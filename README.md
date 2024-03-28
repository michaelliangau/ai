# AI

Monorepo for all AI/deep learning related code intended purely for educational and experimental purposes.

## Projects

All project specific code exists in the `projects` folder. These projects are ordered in chronologically.
- `audio_classification_freesound` - Freesound audio classification Kaggle challenge 2019.
- `asr_finetune_w2v2` - English ASR by finetuning Meta's Wav2Vec2 on various datasets.
- `tts_tacotron2` - Text to speech TacoTron 2.
- `rl_mad_mario` - Reinforcement learning to train an agent in Mario game.
- `np_rnn` - Build a RNN using numpy.
- `np_cnn` - Build a CNN using numpy.
- `pytorch_transformer` - Build a transformer using PyTorch.
- `binary_classification_spaceship_titanic` - Binary classification Kaggle competition.
- `predict_house_price` - Housing price prediction Kaggle competition.
- `gan_mnist` - Generative Adversial Networks using the MNIST dataset.
- `gan_monet` - Generative Adversial Networks Kaggle competition.
- `gan_pickle` - CycleGAN to turn merge any picture with the cutest dog.
- `stable_diffusion_2` - Stable diffusion 2 project.
- `web_scrapers` - Various web scraping projects.
- `micrograd` - Backpropagation and neural network primitives.
- `makemore` - Bigram and neural network language model primitive.
- `gpt` - Transformer based character-level language model.
- `forward_forward` - Geoffrey Hinton's proposed forward forward neural network training algorithm.
- `image_diffusion` - Text to Image diffusion primitives
- `rnnt` - Recurrent neural network transducer architecture for seq2seq problems.
- `buffet_bot` - Algorithmic investing using LLMs
- `silicron` - PyPI package to extend chat apps with context using vector stores.
- `rl_primitives` - RL primitives, implement agents in grid world environment.
- `gpt2_shadow` - Use RL to finetune an LLM to adversarially train against an AI text detector.
- `entity_categorisation` - Assign entity descriptions into a set of categories.
- `lao_asr_s2tt` - Improve Lao ASR and S2TT.
- `librispeech_data_processing` - Create word-level timestamp annotated datasets.

## Common utils

`common` contains common utils that are used across projects.

```python
# Import packages from parent directories
import sys
sys.path.append("../..")
import common.utils as common_utils
```

## Training run logging

```python
common_utils.start_wandb_logging(project_name="ai", config_dict={})
```

## Formatting code

```bash
make format
```
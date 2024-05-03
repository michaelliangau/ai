# Computer Agent

## Usage

Start Chrome in docker container
```bash
docker run -d -p 4444:4444 --shm-size="2g" selenium/standalone-chrome
```

Run script
```bash
python main.py
```

## Findings

FUYU can come up with a plan and find the location in the image to interact with.

From playing around with Fuyu, you can get somewhat reliable box boundings to text if you make sure the resolution of your image going in in 1920 x 1080. It seems the model was trained on images of this resolution and it is not reliable otherwise. I'm unclear if LLAVA and QWEN have similar dynamics but naively I didn't get any good results from them.

FUYU, QWEN and LLAVA all seem to have decent and similar level of multimodal understanding.

RCI was helpful in refining the quality of the plan. However I found that the model tends to continue making mistakes if a mistake was made in the past (past context) so it's probably important to have some sort of reset mechanism.

### MVP findings - Click the red square

Ask a model to click on a red square in a 1920 x 1080 image using just a simple ResNet50 image encoder followed by MLP.

[WNB run](https://wandb.ai/michaelliangaus/huggingface/runs/6562wtuh)

Next: Incorporate text encodings and distinguish between a red square and a yellow square.



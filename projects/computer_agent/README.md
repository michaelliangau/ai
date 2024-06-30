# Computer Agent

Build a computer action agent that is able to complete web forms.

## Milestones
- ✅ Get a VLLM model to locate a dynamically generated object ("Give me x,y coordinates of the red square")
- ✅ Get a VLLM model to dynamically locate a 1 of 3 dynamically generated object ("Give me x,y coordinates of the red/blue/green square")
- ✅ Get a VLLM model to dynamically locate a 1 of 3 dynamically generated object in a multi-step changing environment (3 serial instructions of "Give me x,y coordinates of the red/blue/green square" where after each step the chosen square is removed in a new image)
- ✅ Get a VLLM model to overfit to form completion actions on a single page form screenshot ("Click the name box" --> "Type Michael" --> Click the email box" --> etc)
- 🛠️ Get a VLLM model to dynamically perform form completion actions on a dynamically generated form ("Click the name box" --> "Type Michael" --> Click the email box" --> etc)

## Usage

Start Chrome in docker container
```bash
docker run -d -p 4444:4444 --shm-size="2g" selenium/standalone-chrome
```

Run script
```bash
python main.py
```

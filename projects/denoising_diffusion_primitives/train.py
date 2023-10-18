import datasets
import transformers
import torchvision
import model

# Hyperparameters
forward_beta = 0.2
forward_num_timesteps = 100
forward_decay_rate = 0.98

# Tokenizer
tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
text_embedding_model = transformers.T5EncoderModel.from_pretrained("t5-small")
text = "Hello my name is Michale"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = text_embedding_model(**inputs)
text_embedding = outputs.last_hidden_state
average_text_embedding = text_embedding.mean(dim=1)

# TODO: This is just toy code above tokenizer stuff.

# Data
ds = datasets.load_dataset('HuggingFaceM4/COCO', '2014_captions', split='test') # TODO: Don't use the test split
transform = torchvision.transforms.ToTensor()
image = transform(ds[0]["image"])
text = ds[0]["sentences_raw"][0]

# Data Loader


# Forward Noising Step
forward_process = model.ForwardProcess(num_timesteps=forward_num_timesteps, initial_beta=forward_beta, decay_rate=forward_decay_rate)
noised_image = forward_process.sample(image=image, timestep=0)

# Backward Generation Step
backward_process = model.BackwardProcess()
denoised_image = backward_process.denoise(image=noised_image, text=text, timestep=0)
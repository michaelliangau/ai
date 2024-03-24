from datasets import load_dataset

# Load the dataset from Hugging Face Datasets
dataset = load_dataset("olympusmons/librispeech_asr_test_clean_word_timestamp")
print("Dataset successfully loaded.")
print(f"Dataset features: {dataset.column_names}")
print(f"Number of samples: {len(dataset['train'])}")
import IPython; IPython.embed()


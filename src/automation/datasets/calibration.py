from automation.datasets.utils import load_dataset_messages

DATASET_PATH = "neuralmagic/LLM_compression_calibration"

def load_calibration_dataset(num_samples=None, max_seq_len=None, tokenizer=None):
    return load_dataset_messages(DATASET_PATH, split="train", num_samples=num_samples, max_seq_len=max_seq_len, tokenizer=tokenizer)

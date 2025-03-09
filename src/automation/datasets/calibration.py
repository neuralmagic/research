from datasets import load_dataset

DATASET_PATH = "neuralmagic/LLM_compression_calibration"

def load_calibration_dataset(num_samples=None, max_seq_len=None, tokenizer=None):

    ds = load_dataset(DATASET_PATH, split="train")

    if tokenizer is not None:
        def preprocess_calibration(example):
            if tokenizer.chat_template is None:
                return {"text": example["text"]}
            else:
                try:
                    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
                except:
                    messages = example["messages"][1:]
                    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

        ds = ds.map(preprocess_calibration, remove_columns=ds.column_names)
        ds = ds.map(lambda example: tokenizer(example["text"], padding=False, max_length=max_seq_len, truncation=True), remove_columns=ds.column_names)
    
        if "token_type_ids" in ds.features:
            ds = ds.remove_columns(["token_type_ids"])

    return ds
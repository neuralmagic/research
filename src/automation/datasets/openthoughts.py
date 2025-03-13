from datasets import load_dataset

DATASET_PATH = "open-thoughts/OpenThoughts-114k"
PERCENTAGE = 5

def load_openthoughts_dataset(num_samples=None, max_seq_len=None, tokenizer=None):

    ds = load_dataset(DATASET_PATH, split=f"train[:{PERCENTAGE}%]")

    if tokenizer is not None:
        def preprocess_calibration(example):
            if tokenizer.chat_template is None:
                text = example["system"]
                for conversation in example["conversations"]:
                    text += "\n\n" + conversation["value"]
                return {"text": text}
            else:
                messages = [{"role": "system", "content": example["system"]}]
                for conversation in example["conversations"]:
                    messages.append({"role": conversation["from"], "content": conversation["value"]})
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

        ds = ds.map(preprocess_calibration, remove_columns=ds.column_names)
        ds = ds.map(lambda example: tokenizer(example["text"], padding=False, max_length=max_seq_len, truncation=True), remove_columns=ds.column_names)
    
        if "token_type_ids" in ds.features:
            ds = ds.remove_columns(["token_type_ids"])

    return ds
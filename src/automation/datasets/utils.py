from datasets import load_dataset, interleave_datasets

def load_llm_messages(
    dataset_name, 
    subset=None,
    split="train", 
    num_samples=None, 
    max_seq_len=None, 
    tokenizer=None,
    processor=None,
):

    ds = load_dataset(dataset_name, name=subset, split=split)
    if num_samples is not None:
        ds = ds.select(range(num_samples))

    if tokenizer is None:
        tokenizer = processor

    if tokenizer is not None:
        def preprocess_sample(example):
            if tokenizer.chat_template is None:
                return {"text": example["text"]}
            else:
                try:
                    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
                except:
                    messages = example["messages"][1:]
                    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

        ds = ds.map(preprocess_sample, remove_columns=ds.column_names)
        ds = ds.map(lambda example: tokenizer(example["text"], padding=False, max_length=max_seq_len, truncation=True), remove_columns=ds.column_names)
    
        if "token_type_ids" in ds.features:
            ds = ds.remove_columns(["token_type_ids"])

    return ds


def normalize_content(content):    
    if isinstance(content, str):
        content = [{"type": "text", "text": content, "image": b""}]
    elif isinstance(content, list):
        for i in range(len(content)):
            content[i] = normalize_content(content[i])
    elif isinstance(content, dict):
        if "image" not in content or  content["image"] is None:
            content["image"] = b""
        
        if "text" not in content or content["text"] is None:
            content["text"] = ""
        
    return content


def streamline_content(content):
    from PIL import Image
    import io

    if isinstance(content, list):
        for i in range(len(content)):
            content[i] = streamline_content(content[i])
    elif isinstance(content, dict):
        if len(content["text"]) == 0:
            del content["text"]

        if len(content["image"]) == 0:
            del content["image"]
        else:
            content["image"] = Image.open(io.BytesIO(content["image"]))

       
    return content

def load_vlm_messages(
    dataset_name, 
    subset=None,
    split="train",
    num_samples=None,
    processor=None,
):
    dataset_name = dataset_name if isinstance(dataset_name, list) else [dataset_name]
    subset = subset if isinstance(subset, list) else [subset]
    split = split if isinstance(split, list) else [split]
    num_samples = num_samples if isinstance(num_samples, list) else [num_samples]

    mixing_len = max(len(dataset_name), len(subset), len(split), len(num_samples))
    assert len(dataset_name) == 1 or len(dataset_name) == mixing_len
    assert len(subset) == 1 or len(subset) == mixing_len
    assert len(split) == 1 or len(split) == mixing_len
    assert len(num_samples) == 1 or len(num_samples) == mixing_len

    dataset_name = dataset_name if len(dataset_name) == mixing_len else mixing_len * dataset_name
    subset = subset if len(subset) == mixing_len else mixing_len * subset
    split = split if len(split) == mixing_len else mixing_len * split
    num_samples = num_samples if len(num_samples) == mixing_len else mixing_len * num_samples

    datasets = []
    for dataset_name_, subset_, split_, num_samples_ in zip(dataset_name, subset, split, num_samples):
        print(dataset_name_, subset_, split_, num_samples_)
        if num_samples_ is None or num_samples_ == 0:
            continue

        ds = load_dataset(dataset_name_, name=subset_, split=split_)
        if num_samples_ is not None:
            ds = ds.select(range(num_samples_))

        def align_content(example):
            messages = []
            for message in example["messages"]:
                message["content"] = normalize_content(message["content"])
                messages.append(message)
            
            return {"messages": messages}

        ds = ds.map(align_content, remove_columns=ds.column_names)

        datasets.append(ds)


    dataset = interleave_datasets(datasets)

    def preprocess_sample(example):
        messages = []
        for message in example["messages"]:           
            messages.append(
                {
                    "role": message["role"],
                    "content": streamline_content(message["content"]),
                }
            )
        
        return processor.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )

    return dataset.map(preprocess_sample, remove_columns=ds.column_names)

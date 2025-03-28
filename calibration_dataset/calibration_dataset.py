from datasets import load_dataset
import io


llm_dataset = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
llm_dataset.to_parquet("llm.parquet")

vllm_dataset = load_dataset("lmms-lab/flickr30k", split="test").shuffle().select(range(10000))

def pil_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()


def create_example(example):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_to_bytes(example["image"]),
                },
                {
                    "type": "text", 
                    "text": "What does the image show?",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": example["caption"][0].replace(" .", ".").replace("  ", " "),
                },
            ],
        },    
    ]

    return {"messages": messages}

vllm_dataset = vllm_dataset.map(create_example, remove_columns=vllm_dataset.column_names)
vllm_dataset.to_parquet("vllm.parquet")
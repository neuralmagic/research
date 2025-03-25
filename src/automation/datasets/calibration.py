from automation.datasets.utils import load_llm_messages, load_vllm_messages
from datasets import interleave_datasets

DATASET_PATH = "neuralmagic/calibration"
TEXT_SUBSET = "LLM"
VISION_SUBSET = "VLLM"

def load_calibration_dataset(
    vision_samples=None,
    text_samples=None,
    num_samples=None,
    max_seq_len=None, 
    vision_loader=None,
    tokenizer=None,
    processor=None,
):
    if vision_samples is None and text_samples is None and num_samples is not None:
        text_samples = num_samples

    if vision_samples is None:
        vision_samples = 0

    if text_samples > 0 and vision_samples == 0:
        return load_llm_messages(
            DATASET_PATH, 
            TEXT_SUBSET, 
            split="train", 
            num_samples=text_samples, 
            max_seq_len=max_seq_len, 
            tokenizer=tokenizer,
        )

    if vision_loader is None:
        vision_loader = load_vllm_messages

    vision_dataset = vision_loader(
        DATASET_PATH, 
        VISION_SUBSET, 
        split="train", 
        num_samples=vision_samples,
        processor=processor,
    )

    if text_samples > 0:
        text_dataset = vision_loader(
            DATASET_PATH, 
            TEXT_SUBSET, 
            split="train", 
            num_samples=text_samples,
            processor=processor,
        )

        return interleave_datasets([vision_dataset, text_dataset])

    return vision_dataset

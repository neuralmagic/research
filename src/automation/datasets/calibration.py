from automation.datasets.utils import load_text_dataset, load_vision_dataset, mix_datasets

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
    if tokenizer is None:
        tokenizer = processor
        
    if vision_samples is None and text_samples is None and num_samples is not None:
        text_samples = num_samples

    if text_samples is not None and text_samples > 0:
        text_dataset = load_text_dataset(
            DATASET_PATH, 
            TEXT_SUBSET, 
            split="train", 
            num_samples=text_samples, 
            max_seq_len=max_seq_len, 
            tokenizer=tokenizer,
        )

    if vision_samples is not None and vision_samples > 0:
        if vision_loader is None:
            vision_dataset = load_text_dataset(
                DATASET_PATH, 
                TEXT_SUBSET, 
                split="train", 
                num_samples=text_samples, 
                processor=processor,
            )
        else:
            vision_dataset = vision_loader(
                DATASET_PATH, 
                VISION_SUBSET, 
                split="train", 
                num_samples=vision_samples,
                processor=processor,
            )

    if vision_samples > 0 and text_samples > 0:
        return mix_datasets(vision_dataset, text_dataset)
    elif vision_samples > 0:
        return vision_dataset
    else:
        return text_dataset
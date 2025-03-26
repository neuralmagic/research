from automation.datasets.utils import load_llm_messages, load_vlm_messages

DATASET_PATH = "neuralmagic/calibration"
TEXT_SUBSET = "LLM"
VISION_SUBSET = "VLM"

def load_calibration_dataset(
    vision_samples=None,
    text_samples=None,
    num_samples=None,
    max_seq_len=None, 
    multimodal_loader=None,
    tokenizer=None,
    processor=None,
):
    if vision_samples is None and text_samples is None and num_samples is not None:
        text_samples = num_samples

    if text_samples > 0 and vision_samples == 0:
        return load_llm_messages(
            DATASET_PATH, 
            TEXT_SUBSET, 
            split="train", 
            num_samples=text_samples, 
            max_seq_len=max_seq_len, 
            tokenizer=tokenizer,
        )

    if multimodal_loader is None:
        multimodal_loader = load_vlm_messages

    return multimodal_loader(
        dataset_name=DATASET_PATH, 
        subset=[TEXT_SUBSET, VISION_SUBSET], 
        split="train", 
        num_samples=[text_samples, vision_samples],
        processor=processor,
    )
from automation.datasets.utils import load_llm_messages, load_vlm_messages

DATASET_PATH = "neuralmagic/calibration"
TEXT_SUBSET = "LLM"
VISION_SUBSET = "VLM"

def load_calibration_dataset(
    vision_samples=None,
    text_samples=None,
    max_seq_len=None, 
    multimodal_loader=None,
    processor=None,
):
    if text_samples is None:
        text_samples = 0
    
    if vision_samples is None:
        vision_samples = 0
        
    if text_samples > 0 and vision_samples == 0:
        return load_llm_messages(
            DATASET_PATH, 
            TEXT_SUBSET, 
            split="train", 
            num_samples=text_samples, 
            max_seq_len=max_seq_len, 
            tokenizer=processor,
        )

    if multimodal_loader is None:
        multimodal_loader = load_vlm_messages

    return multimodal_loader(
        dataset_name=DATASET_PATH, 
        subset=[VISION_SUBSET, TEXT_SUBSET], 
        split="train", 
        num_samples=[vision_samples, text_samples],
        processor=processor,
    )
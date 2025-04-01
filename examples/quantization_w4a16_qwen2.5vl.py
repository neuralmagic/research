from automation.tasks import LLMCompressorTask

recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ["lm_head", "re:visual.*"]
      sequential_targets: ["Qwen2_5_VLDecoderLayer"]
      dampening_frac: 0.01
      config_groups:
        group0:
          targets: ["Linear"]
          weights:
            num_bits: 4
            type: "int"
            strategy: "group"
            group_size: 128
            symmetric: true
            actorder: "weight"
            observer: "mse"
"""

def dataset_loader(
    dataset_name, 
    vision_samples, 
    text_samples, 
    max_seq_len,
    processor,
):
    from automation.datasets import load_vlm_messages
    from automation.datasets.calibration import DATASET_PATH, TEXT_SUBSET, VISION_SUBSET
    from qwen_vl_utils import process_vision_info
    import base64
    from io import BytesIO

    def message_processor(messages, processor):
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    buffered = BytesIO()
                    content["image"].save(buffered, format="PNG")
                    encoded_image = base64.b64encode(buffered.getvalue())
                    encoded_image_text = encoded_image.decode("utf-8")
                    base64_qwen = f"data:image;base64,{encoded_image_text}"
                    content["image"] = base64_qwen

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=max_seq_len,
            truncation=True,
        )
    
    return load_vlm_messages(
        DATASET_PATH, 
        [TEXT_SUBSET, VISION_SUBSET], 
        num_samples=[text_samples, vision_samples], 
        processor=processor,
        message_processor=message_processor,
    )

def data_collator(batch):
    import torch
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

task = LLMCompressorTask(
    project_name="Qwen2.5-VL/W4A16",
    task_name="Qwen2.5-VL-7B-Instruct-quantized.w4a16/damp=0.01/actorder=weight/observer=mse/text_samples=512/vision_samples=512",
    packages=["qwen-vl-utils", "torchvision"],
    recipe=recipe,
    dataset_loader=dataset_loader,
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    tracing_class="TraceableQwen2_5_VLForConditionalGeneration",
    trust_remote_code=True,
    text_samples=512,
    vision_samples=512,
    max_memory_per_gpu="auto",
    data_collator=data_collator,
    max_seq_len=2048,
)


task.execute_remotely("oneshot-a100x1")

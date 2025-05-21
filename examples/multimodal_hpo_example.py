from automation.pipelines import LLMCompressorLMEvalPipeline
from automation.hpo import BaseHPO
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange

recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ["language_model.lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"]
      sequential_targets: ["MistralDecoderLayer"]
      dampening_frac: $dampening_frac
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

    def message_processor(messages, processor):
        messages_ = []
        images = None
        for message in messages:
            content_ = []
            for content in message["content"]:
                if content["type"] == "image":
                    content_.append({"type": "image"})
                    images = content["image"]
                else:
                    content_.append(content)
            messages_.append({"role": message["role"], "content": content_})
                  
        input = {
            "text": processor.apply_chat_template(
                messages_,
                add_generation_prompt=False,
            ),
            "images": images,
        }
        tokenized_input = processor(**input, max_length=max_seq_len, truncation=True)
        tokenized_input["pixel_values"] = tokenized_input.get("pixel_values", None)
        tokenized_input["image_sizes"] = tokenized_input.get("image_sizes", None)
        
        tokenized_input = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in tokenized_input.items()
        }

        return tokenized_input
    
    return load_vlm_messages(
        DATASET_PATH, 
        [VISION_SUBSET, TEXT_SUBSET], 
        num_samples=[vision_samples, text_samples], 
        processor=processor,
        message_processor=message_processor,
    )

def data_collator(batch):
    import torch
    assert len(batch) == 1
    collated = {}
    for k, v in batch[0].items():
        if v is None:
            continue
        if k == "input_ids":
            collated[k] = torch.LongTensor(v)
        elif k == "pixel_values":
            collated[k] = torch.tensor(v, dtype=torch.bfloat16)
        else:
            collated[k] = torch.tensor(v)
    return collated

def average_scores(task):
    results = task.get_reported_scalars()
    if "gsm8k" in results:
        gsm8k_score = results["gsm8k"]["exact_match,strict-match"]["y"][0]
    else:
        gsm8k_score = 0.
    if "mmlu" in results:
        mmlu_score = results["mmlu"]["acc,none"]["y"][0]
    else:
        mmlu_score = 0.
    if "mmmu_val" in results:
        mmmu_score = results["mmmu_val"]["acc,none"]["y"][0]
    else:
        mmmu_score = 0.
    if "arc_challenge" in results:
        arc_score = results["arc_challenge"]["acc,none"]["y"][0]
    else:
        arc_score = 0.

    average_score = (gsm8k_score + mmlu_score + mmmu_score + arc_score) / 4.
    task.get_logger().report_scalar(title="score", series="average", iteration=0, value=average_score)


pipeline = LLMCompressorLMEvalPipeline(
    project_name="Mistral/Mistral-Small-3.1-24B-Instruct",
    pipeline_name="Mistral-Small-3_1-24B-Instruct-2503/W4A16/Pipeline",
    model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    execution_queues=["oneshot-a100x2", "oneshot-a100x2", "oneshot-a100x2", "oneshot-a100x2", "oneshot-a100x2"],
    job_end_callback=average_scores, 
    parameters={
        "dampening_frac": {
            "default": 0.01,
            "param_type": "float",
            "recipe_arg": True,
        },
        "text_samples": {
            "default": 512,
            "param_type": "int",
            "recipe_arg": False,
        },
        "vision_samples": {
            "default": 512,
            "param_type": "int",
            "recipe_arg": False,
        },
    },
    llmcompressor_kwargs={
        "recipe": recipe, 
        "dataset_loader": dataset_loader,
        "data_collator": data_collator,
        "tracing_class": "TraceableMistral3ForConditionalGeneration",
        "model_class": "AutoModelForImageTextToText",
        "max_seq_len": 8192,
        "max_memory_per_gpu": "auto",
    },
    lmeval_kwargs={
        "evaluation_mmlu": {
            "tasks": "mmlu",
            "monitor_metrics": [["mmlu", "acc,none"]],
            "model_args": "gpu_memory_utilization=0.5,enable_chunked_prefill=True,max_model_len=8192", 
            "apply_chat_template": True, 
            "fewshot_as_multiturn": True,
            "num_fewshot": 5,
            "batch_size": "auto", 
            "packages": ["numpy==2.1"],
        },
        "evaluation_gsm8k": {
            "tasks": "gsm8k",
            "monitor_metrics": [["gsm8k", "exact_match,strict-match"]],
            "model_args": "gpu_memory_utilization=0.9,enable_chunked_prefill=True,max_model_len=8192", 
            "apply_chat_template": True, 
            "fewshot_as_multiturn": True,
            "num_fewshot": 5,
            "batch_size": "auto", 
            "packages": ["numpy==2.1"],
        },
        "evaluation_arc": {
            "tasks": "arc_challenge",
            "monitor_metrics": [["arc_challenge", "acc,none"]],
            "model_args": "gpu_memory_utilization=0.5,enable_chunked_prefill=True,max_model_len=8192", 
            "apply_chat_template": True, 
            "fewshot_as_multiturn": True,
            "num_fewshot": 25,
            "batch_size": "auto", 
            "packages": ["numpy==2.1"],
        },
        "evaluation_mmmu": {
            "model": "vllm-vlm",
            "tasks": "mmmu_val",
            "monitor_metrics": [["mmmu_val", "acc,none"]],
            "model_args": "gpu_memory_utilization=0.5,enable_chunked_prefill=True,max_model_len=8192,max_images=8", 
            "apply_chat_template": True, 
            "batch_size": "auto", 
            "packages": ["numpy==2.1"],
        },
    },
)

pipeline.create_pipeline()

hpo_task = BaseHPO(
    project_name="Mistral/Mistral-Small-3.1-24B-Instruct",
    task_name="Mistral-Small-3_1-24B-Instruct-2503/W4A16/HPO",
    report_period_min=5,
    optimizer="Optuna",
    optuna_sampler="TPESampler",
    optuna_sampler_kwargs={"n_startup_trials": 4},
    objective_metric_title="score",
    objective_metric_series="average",
    objective_metric_sign="max",
    total_max_jobs=20,
    max_number_of_concurrent_tasks=4,
    pool_period_min=1,
    max_iteration_per_job=1,
    spawn_project="Mistral/Mistral-Small-3.1-24B-Instruct/W4A16_hpo",
    base_task_id= pipeline.id,
)

hpo_task.add_parameter(UniformParameterRange("Args/dampening_frac", min_value=0.01, max_value=0.2))
hpo_task.add_parameter(UniformIntegerParameterRange("Args/text_samples", min_value=512, max_value=1024, step_size=512))
hpo_task.add_parameter(UniformIntegerParameterRange("Args/vision_samples", min_value=512, max_value=1024, step_size=512))

hpo_task.execute_remotely()

import dill
import os
from automation.datasets import SUPPORTED_DATASETS
from automation.standards.compression.smoothquant_mappings import MAPPINGS_PER_MODEL_CONFIG
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoProcessor
from clearml import OutputModel, Task
import torch
from automation.utils import resolve_model_id, parse_argument
from llmcompressor.transformers import tracing


def main(
    model_id,
    clearml_model,
    force_download,
    max_memory_per_gpu,
    tracing_class,
    trust_remote_code,
    recipe,
    recipe_args,
    dataset_loader,
    dataset_name,
    max_seq_len,
    text_samples,
    vision_samples,
    save_directory,
    data_collator,
    tags,
    task,
):
    # Resolve model_id
    model_id = resolve_model_id(model_id, clearml_model, force_download)

    # Set dtype
    dtype = "auto"

    # Set device map
    if max_memory_per_gpu == "auto":
        device_map = "auto"
    else:
        # Determine number of gpus
        num_gpus = torch.cuda.device_count()
        
        if max_memory_per_gpu == "hessian":
            device_map = calculate_offload_device_map(
                model_id, 
                reserve_for_hessians=True, 
                num_gpus=num_gpus, 
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            device_map = custom_offload_device_map(
                model_id, 
                max_memory_per_gpu=max_memory_per_gpu + "GB",
                num_gpus=num_gpus, 
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    # Load model
    if tracing_class is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map=device_map, 
            trust_remote_code=trust_remote_code,
        )
    else:
        model_class = getattr(tracing, tracing_class)
        model = model_class.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map=device_map, 
            trust_remote_code=trust_remote_code,
        )

    # Load recipe
    if isinstance(recipe, str) and os.path.isfile(recipe):
        with open(recipe, "r", encoding="utf-8") as file:
            recipe = file.read()

    if recipe_args is not None:
        if "smoothquant_mappings" in recipe_args and recipe_args["smoothquant_mappings"] in MAPPINGS_PER_MODEL_CONFIG:
            recipe_args["smoothquant_mappings"] = MAPPINGS_PER_MODEL_CONFIG[recipe_args["smoothquant_mappings"]]
            
        for key, value in recipe_args.items():
            recipe = recipe.replace(f"${key}", str(value))

    if task is not None:
        task.upload_artifact("recipe", recipe)
        
    # Load dataset
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=trust_remote_code,
    )

    print("before dataset")

    if dataset_loader is None:
        if dataset_name is None:
            dataset = None
        elif dataset_name in SUPPORTED_DATASETS:
            dataset = SUPPORTED_DATASETS[dataset_name](
                text_samples=text_samples,
                vision_samples=vision_samples,
                max_seq_len=max_seq_len,
                processor=processor,
            )
    else:
        dataset = dataset_loader(
            dataset_name,
            text_samples=text_samples,
            vision_samples=vision_samples,
            max_seq_len=max_seq_len,
            processor=processor,
        )
    
    print("after dataset")

    num_calibration_samples = 0
    if text_samples is not None:
        num_calibration_samples += text_samples

    if vision_samples is not None:
        num_calibration_samples += vision_samples

    kwargs = {}
    if data_collator is not None:
        kwargs["data_collator"] = data_collator

    # Apply recipe to the model
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_calibration_samples,
        **kwargs,
    )

    # Save model compressed
    model.save_pretrained(save_directory, save_compressed=True)
    processor.save_pretrained(save_directory)

    # Upload model to ClearML
    if task is not None:
        clearml_model = OutputModel(
            task=task, 
            name=task.name,
            framework="PyTorch", 
            tags=[tags] if isinstance(tags, str) else tags or []
        )
        clearml_model.update_weights(weights_filename=save_directory, auto_delete_file=False)


if __name__ == '__main__':
    task = Task.current_task()

    # Parse arguments
    args = task.get_parameters_as_dict(cast=True)["Args"]
    clearml_model = parse_argument(args["clearml_model"], bool)
    force_download = parse_argument(args["force_download"], bool)
    trust_remote_code = parse_argument(args["trust_remote_code"], bool)
    model_id = parse_argument(args["model_id"], str)
    dataset_name = parse_argument(args["dataset_name"], str)
    tracing_class = parse_argument(args["tracing_class"], str)
    save_directory = parse_argument(args["save_directory"], str)
    max_memory_per_gpu = parse_argument(args["max_memory_per_gpu"], str)
    max_seq_len = parse_argument(args["max_seq_len"], int)
    text_samples = parse_argument(args["text_samples"], int)
    vision_samples = parse_argument(args["vision_samples"], int)
    recipe = args.get("recipe", None)
    recipe_args = args.get("recipe_args", None)
    tags = args.get("tags", None)

    dataset_loader = parse_argument(args["dataset_loader"], str)
    if dataset_loader is not None:
        dataset_loader = dill.load(task.artifacts[dataset_loader].get())

    data_collator = parse_argument(args["data_collator"], str)
    if data_collator is not None:
        data_collator = dill.load(task.artifacts[data_collator].get())
        print(data_collator)


    main(
        model_id,
        clearml_model,
        force_download,
        max_memory_per_gpu,
        tracing_class,
        trust_remote_code,
        recipe,
        recipe_args,
        dataset_loader,
        dataset_name,
        max_seq_len,
        text_samples,
        vision_samples,
        save_directory,
        data_collator,
        tags,
        task,
    )
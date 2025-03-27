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
from automation.utils import resolve_model_id
from llmcompressor.transformers import tracing

def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)["Args"]
    clearml_model = args["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"

    force_download = args["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    trust_remote_code = args["trust_remote_code"]
    if isinstance(trust_remote_code, str):
        trust_remote_code = trust_remote_code.lower() == "true"

    dataset_name = args["dataset_name"]
    if isinstance(dataset_name, str) and dataset_name.lower() == "none":
        dataset_name = None

    dataset_loader = args["dataset_loader"]
    if isinstance(dataset_loader, str) and dataset_loader.lower() == "none":
        dataset_loader = None

    tracing_class = args["tracing_class"]
    if isinstance(tracing_class, str) and tracing_class.lower() == "none":
        tracing_class = None

    max_seq_len = int(args["max_seq_len"])

    num_samples = args["num_samples"]
    if isinstance(num_samples, str):
        if num_samples.lower() == "none":
            num_samples = None
    num_samples = int(num_samples)

    text_samples = args["text_samples"]
    if isinstance(text_samples, str):
        if text_samples.lower() == "none":
            text_samples = None
    text_samples = int(text_samples)

    vision_samples = args["vision_samples"]
    if isinstance(vision_samples, str):
        if vision_samples.lower() == "none":
            vision_samples = None
    vision_samples = int(vision_samples)


    # Resolve model_id
    model_id = resolve_model_id(args["model_id"], clearml_model, force_download)

    # Set dtype
    dtype = "auto"

    # Set device map
    if args["max_memory_per_gpu"] == "auto":
        device_map = "auto"
    else:
        # Determine number of gpus
        num_gpus = torch.cuda.device_count()
        
        if args["max_memory_per_gpu"] == "hessian":
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
                max_memory_per_gpu=args["max_memory_per_gpu"] + "GB",
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
    recipe = args["recipe"]
    if isinstance(recipe, str) and os.path.isfile(recipe):
        with open(recipe, "r", encoding="utf-8") as file:
            recipe = file.read()

    recipe_args = args.get("recipe_args", None)
    if recipe_args is not None:
        if "smoothquant_mappings" in recipe_args and recipe_args["smoothquant_mappings"] in MAPPINGS_PER_MODEL_CONFIG:
            recipe_args["smoothquant_mappings"] = MAPPINGS_PER_MODEL_CONFIG[recipe_args["smoothquant_mappings"]]
            
        for key, value in args["recipe_args"].items():
            recipe = recipe.replace(f"${key}", str(value))

    task.upload_artifact("recipe", recipe)
        
    # Load dataset
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=trust_remote_code,
    )

    if dataset_loader is None:
        if dataset_name is None:
            dataset = None
        elif args["dataset_name"] in SUPPORTED_DATASETS:
            dataset = SUPPORTED_DATASETS[args["dataset_name"]](
                text_samples=text_samples,
                vision_samples=vision_samples,
                num_samples=num_samples,
                max_seq_len=max_seq_len,
                processor=processor,
            )
    else:
        dataset_loader_path = task.artifacts[dataset_loader].get_local_copy()
        dataset_loader = dill.load(open(dataset_loader_path, "rb"))
        dataset = dataset_loader(
            args["dataset_name"],
            text_samples=text_samples,
            vision_samples=vision_samples,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            processor=processor,
        )


    # Apply recipe to the model
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples,
    )

    # Save model compressed
    model.save_pretrained(args["save_directory"], save_compressed=True)
    processor.save_pretrained(args["save_directory"])

    # Upload model to ClearML
    clearml_model = OutputModel(
        task=task, 
        name=task.name,
        framework="PyTorch", 
        tags=[args["tags"]] if isinstance(args["tags"], str) else args["tags"] or []
    )
    clearml_model.update_weights(weights_filename=args["save_directory"], auto_delete_file=False)


if __name__ == '__main__':
    main()
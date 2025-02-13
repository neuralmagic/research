import os
from automation.datasets import load_calibration_dataset, CALIBRATION_DATASET
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM
from clearml import OutputModel, Task
import torch
from automation.utils import resolve_model_id

def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)["Args"]

    # Resolve model_id
    model_id = resolve_model_id(args["model_id"], args["clearml_model"], task)

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
                trust_remote_code=args["trust_remote_code"],
            )
        else:
            device_map = custom_offload_device_map(
                model_id, 
                max_memory_per_gpu=args["max_memory_per_gpu"] + "GB",
                num_gpus=num_gpus, 
                torch_dtype=dtype,
                trust_remote_code=args["trust_remote_code"],
            )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map=device_map, 
        trust_remote_code=args["trust_remote_code"],
    )

    # Load recipe
    recipe = args["recipe"]
    if not isinstance(recipe, str):
        if os.path.isfile(recipe):
            with open(recipe, "r", encoding="utf-8") as file:
                recipe = file.read()

    if args["recipe_args"] is not None:
        for key, value in args["recipe_args"].items():
            recipe = recipe.replace(f"${key}", str(value))

    task.upload_artifact("recipe", recipe)
        
    # Load dataset
    if args["dataset_name"] in ["calibration", CALIBRATION_DATASET]:
        dataset = load_calibration_dataset()
    else:
        raise ValueError("Dataset not supported.")

    # Apply recipe to the model
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args["max_seq_len"],
        num_calibration_samples=args["num_samples"],
    )

    # Save model compressed
    model.save_pretrained(args["save_directory"], save_compressed=True)

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
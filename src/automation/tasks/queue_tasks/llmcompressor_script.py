import os
from automation.datasets import load_calibration_dataset, CALIBRATION_DATASET
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM
from clearml import InputModel, OutputModel, Task
import torch

task = Task.get_current_task()

args = task.get_parameters_as_dict()

# Resolve model_id
if args.clearml_model:
    input_model = InputModel(model_id=args.model_id)
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args.model_id

# Set dtype
if args.dtype == "auto":
    dtype = "auto"
else:
    dtype = getattr(torch, args.dtype)

# Set device map
if args.max_memory_per_gpu == "auto":
    device_map = "auto"
else:
    # Determine number of gpus
    user_properties = task.get_user_properties()
    queue_name = user_properties["k8s-queue"]["value"]
    
    if "single" in queue_name or "x1" in queue_name:
        num_gpus = 1
    elif "double" in queue_name or "x2" in queue_name:
        num_gpus = 2
    elif "quad" in queue_name or "x4" in queue_name:
        num_gpus = 4
    elif "octo" in queue_name or "x8" in queue_name:
        num_gpus = 8
    
    if args.max_memory_per_gpu == "hessian":
        device_map = calculate_offload_device_map(
            model_id, 
            reserve_for_hessians=True, 
            num_gpus=num_gpus, 
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        device_map = custom_offload_device_map(
            model_id, 
            max_memory_per_gpu=args.max_memory_per_gpu + "GB",
            num_gpus=num_gpus, 
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=dtype, 
    device_map=device_map, 
    trust_remote_code=args.trust_remote_code,
)

# Load recipe
recipe = args.recipe    
if not isinstance(recipe, str):
    if os.path.isfile(recipe):
        with open(recipe, "r", encoding="utf-8") as file:
            recipe = file.read()
    
# Load dataset
if args.dataset_name in ["calibration", CALIBRATION_DATASET]:
    dataset = load_calibration_dataset()
else:
    raise ValueError("Dataset not supported.")

# Apply recipe to the model
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=args.max_seq_len,
    num_calibration_samples=args.num_samples,
)

# Save model compressed
model.save_pretrained(args.save_dir, save_compressed=True)

# Upload model to ClearML
clearml_model = OutputModel(
    task=task, 
    name=task.name,
    framework="PyTorch", 
    tags=[args.tags] if isinstance(args.tags, str) else args.tags or []
)
clearml_model.update_weights(weights_filename=args.save_dir, auto_delete_file=False)
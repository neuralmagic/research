import json
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

#from automation.utils import resolve_reference_model_id, parse_argument, load_callable_configuration

try:
    from clearml import OutputModel, Task
    clearml_available = True
except ImportError:
    clearml_available = False

"""
def llmcompressor_main(
    reference_model_id,
    model_class,
    trust_remote_code,
    recipe,
    dataset_args,
    dataset_loader,
    dataset_name,
    max_model_len,
    num_samples,
    max_new_tokens,
    skip_sparsity_compression_stats,
    save_directory,
    data_collator,
):
    dtype = "auto"
    device_map = "auto"

    # Load model
    model_class = getattr(transformers, model_class)

    model = model_class.from_pretrained(
        reference_model_id, 
        torch_dtype=dtype, 
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
    )

    # Load recipe
    if isinstance(recipe, str) and os.path.isfile(recipe):
        with open(recipe, "r", encoding="utf-8") as file:
            recipe = file.read()

    if dataset_args is not None:
        if "smoothquant_mappings" in dataset_args and dataset_args["smoothquant_mappings"] in MAPPINGS_PER_MODEL_CONFIG:
            dataset_args["smoothquant_mappings"] = MAPPINGS_PER_MODEL_CONFIG[dataset_args["smoothquant_mappings"]]
            
        for key, value in dataset_args.items():
            recipe = recipe.replace(f"${key}", str(value))
        
    # Load dataset
    processor = AutoProcessor.from_pretrained(
        reference_model_id, 
        trust_remote_code=trust_remote_code,
    )

    if dataset_loader is None:
        if dataset_name is None:
            dataset = None
        elif dataset_name in SUPPORTED_DATASETS:
            dataset = SUPPORTED_DATASETS[dataset_name](
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                max_model_len=max_model_len,
                processor=processor,
            )
    else:
        dataset = dataset_loader(
            dataset_name,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            processor=processor,
        )
    
    num_calibration_samples = 0
    if num_samples is not None:
        num_calibration_samples += num_samples

    if max_new_tokens is not None:
        num_calibration_samples += max_new_tokens

    kwargs = {}
    if data_collator is not None:
        kwargs["data_collator"] = data_collator

    # Apply recipe to the model
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_model_length=max_model_len,
        num_calibration_samples=num_calibration_samples,
        **kwargs,
    )

    # Save model compressed
    model.save_pretrained(save_directory, save_compressed=True, skip_sparsity_compression_stats=skip_sparsity_compression_stats)
    processor.save_pretrained(save_directory)

    return recipe

"""

def main(configurations=None, args=None):
    if clearml_available:
        task = Task.current_task()
        args = task.get_parameters_as_dict(cast=True)["Args"]
    else:
        args = args["Args"]

    # Parse arguments
    clearml_model = parse_argument(args["clearml_model"], bool)
    force_download = parse_argument(args["force_download"], bool)
    trust_remote_code = parse_argument(args["trust_remote_code"], bool)
    reference_model_id = parse_argument(args["reference_model_id"], str)
    candidate_model_id= parse_argument(args["candidate_model_id"], str)
    dataset_name = parse_argument(args["dataset_name"], str)
    save_directory = parse_argument(args["save_directory"], str)
    max_model_len = parse_argument(args["max_model_len"], int)
    num_samples = parse_argument(args["num_samples"], int)
    max_new_tokens = parse_argument(args["max_new_tokens"], int)
    dataset_args = args.get("dataset_args", None)
    tags = args.get("tags", None)

    """

    dataset_loader_fn = load_callable_configuration("dataset loader", configurations)
    data_collator_fn = load_callable_configuration("data collator", configurations)

    # Resolve reference_model_id
    reference_model_id = resolve_reference_model_id(reference_model_id, clearml_model, force_download, model_class)

    recipe = llmcompressor_main(
        reference_model_id,
        model_class,
        trust_remote_code,
        recipe,
        dataset_args,
        dataset_loader_fn,
        dataset_name,
        max_model_len,
        num_samples,
        max_new_tokens,
        skip_sparsity_compression_stats,
        save_directory,
        data_collator_fn,
    )


    if clearml_available:
        task.upload_artifact("recipe", recipe)

        # Upload model to ClearML
        clearml_model_object = OutputModel(
            task=task, 
            name=task.name,
            framework="PyTorch", 
           tags=[tags] if isinstance(tags, str) else tags or []
        )
        clearml_model_object.update_weights(weights_filename=save_directory, auto_delete_file=False)
    """

if __name__ == '__main__':
    main()

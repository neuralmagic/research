from automation.vllm import VLLMServer

import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

# Import the FLEURS dataset script
from automation.datasets import load_fleurs_dataset
from automation.datasets.fleurs import _FLEURS_LANG_TO_ID, _FLEURS_LANG_SHORT_TO_LONG, _FLEURS_LONG_TO_LANG
from automation.requests import SUPPORTED_REQUESTS
from automation.metrics import WERMetric
from automation.requests.mistral.transcript import transcript_request
from automation.utils import resolve_model_id, cast_args, load_callable_configuration
from pyhocon import ConfigFactory

try:
    from clearml import Task
    clearml_available = True
except ImportError:
    print("ClearML is not installed.")
    clearml_available = False


def fleurs_main(
    model_id,
    language,
    vllm_args={},
    target="http://localhost:8000/v1",
    server_wait_time=120,
    transcript_request_fn=transcript_request,
    temperature=0.0,
    max_concurrent_requests=64,
):

    # Load the dataset
    fleurs_samples = load_fleurs_dataset(name=_FLEURS_LONG_TO_LANG[_FLEURS_LANG_SHORT_TO_LONG[language]], split="test")

    # Start vLLM server
    vllm_server = VLLMServer(
        vllm_args=vllm_args,
        model_id=model_id,
        target=target,
        server_wait_time=server_wait_time,
    )
    vllm_server.start()

    # Initialize the metrics
    wer_metric = WERMetric(language=language)

    # Initialize the client
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=target,
    )

    # Initialize the semaphore (number of concurrent requests)
    semaphore = asyncio.Semaphore(max_concurrent_requests)


    async def process_sample(sample: dict):
        async with semaphore:
            request = transcript_request_fn(
                sample, 
                model_id=model_id, 
                temperature=temperature, 
                language=_FLEURS_LANG_TO_ID[sample["language"]],
            )
            response = await client.audio.transcriptions.create(**request)
        wer_metric(sample["transcription"], response.text)

    async def process_samples():
        # Create a progress bar
        pbar = tqdm(total=len(fleurs_samples), desc="Processing FLEURS samples")
        
        # Process samples with progress tracking
        async def process_sample_with_progress(sample: dict):
            result = await process_sample(sample)
            pbar.update(1)
            return result
        
        tasks = [process_sample_with_progress(sample) for sample in fleurs_samples]
        await asyncio.gather(*tasks)
        
        # Close the progress bar
        pbar.close()

    asyncio.run(process_samples())
    vllm_server.stop()

    print(f"WER mean: {wer_metric.mean()}")
    print(f"WER std: {wer_metric.std()}")

    return wer_metric


def main(configurations, args):

    if clearml_available:
        task = Task.current_task()
    else:
        task = None

    if task is not None and args is None:
        args = task.get_parameters_as_dict(cast=True)
    
    if task is not None and configurations is None:
        fleurs_args = ConfigFactory.parse_string(task.get_configuration_object("fleurs_args"))
        vllm_args = ConfigFactory.parse_string(task.get_configuration_object("vllm_args"))
        target = task.get_configuration_object("target")
        server_wait_time = task.get_configuration_object("server_wait_time")
    else:
        fleurs_args = configurations.get("fleurs_args", {})
        vllm_args = configurations.get("vllm_args", {})
        target = configurations.get("target")
        server_wait_time = configurations.get("server_wait_time", 120)


    model_id = args["Args"]["model_id"]
    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"
    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(model_id, clearml_model, force_download)
    print

    if isinstance(configurations.get("request"), str) and configurations.get("request") in SUPPORTED_REQUESTS:
        request_fn = SUPPORTED_REQUESTS[configurations.get("request")]
    elif callable(configurations.get("request")):
        request_fn = load_callable_configuration("request", configurations)

    results = fleurs_main(
        model_id=model_id,
        vllm_args=vllm_args,
        target=target,
        server_wait_time=server_wait_time,
        transcript_request_fn=request_fn,
        **fleurs_args,
    )

    if task is not None:
        task.upload_artifact(name="results", artifact_object=results)
    return results


if __name__ == "__main__":
    main()
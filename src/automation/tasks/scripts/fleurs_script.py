from automation.vllm import VLLMServer

import asyncio
from openai import AsyncOpenAI

# Import the FLEURS dataset script
from automation.datasets import load_fleurs_dataset
from automation.datasets.fleurs import _FLEURS_LANG_TO_ID
from automation.metrics import WERMetric
from automation.requests.mistral.transcript import transcript_request
from automation.requests.mistral.utils import audio_to_base64

def fleurs_main(
    model_id,
    language,
    vllm_args={},
    server_wait_time=120,
    transcript_request_fn=transcript_request,
    temperature=0.0,
    max_concurrent_requests=64,
):

    # Load the dataset
    fleurs_samples = load_fleurs_dataset(name="en_us", split="test")

    # Start vLLM server
    vllm_server = VLLMServer(
        vllm_args=vllm_args,
        model_id=model_id,
        target="http://localhost:8000/v1",
        server_wait_time=server_wait_time,
    )
    vllm_server.start()

    # Initialize the metrics
    wer_metric = WERMetric(language=language)

    # Initialize the client
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
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
        tasks = [process_sample(sample) for sample in fleurs_samples]
        await asyncio.gather(*tasks)

    asyncio.run(process_samples())

    print(f"WER mean: {wer_metric.mean()}")
    print(f"WER std: {wer_metric.std()}")
    vllm_server.stop()

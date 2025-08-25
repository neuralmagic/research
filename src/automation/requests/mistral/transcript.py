from automation.requests.mistral.utils import audio_to_base64

from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio


def transcript_request(
    sample: dict, 
    model_id:str,
    temperature: float=0.0,
    language: str="en",
):
    audio_base64, audio_format = audio_to_base64(sample["audio"])
    audio = RawAudio(data=audio_base64, format=audio_format.lower())
    return TranscriptionRequest(
        model=model_id, 
        audio=audio, 
        language=language, 
        temperature=temperature).to_openai(exclude=("top_p", "seed"))

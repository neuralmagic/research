from automation.requests.mistral.transcript import transcript_request as mistral_transcript_request

SUPPORTED_REQUESTS = {
    "mistral_transcript_request": mistral_transcript_request,
}

__all__ = [
    "mistral_transcript_request",
    "SUPPORTED_REQUESTS",
]
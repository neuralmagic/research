import io
import base64
import soundfile as sf
import numpy as np
import torch
import os

def audio_to_base64(audio, sample_rate=None):
    """
    Convert audio input to a WAV-encoded Base64 string.
    
    Parameters:
    - audio: 
        * str -> file path to an audio file
        * decoder object with .get_all_samples()
        * NumPy array or PyTorch tensor (raw waveform)
    - sample_rate (int, optional): 
        Required if `audio` is a waveform array/tensor
    
    Returns:
    - str: Base64-encoded WAV audio
    """
    
    # Case 1: File path
    if isinstance(audio, str) and os.path.isfile(audio):
        with open(audio, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")

    # Case 2: Hugging Face decoder object
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        waveform = samples.data.squeeze(0).numpy()
        sr = samples.sample_rate
    
    # Case 3: NumPy array or PyTorch tensor
    elif isinstance(audio, (np.ndarray, torch.Tensor)):
        waveform = audio.squeeze()
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when passing raw waveform.")
        sr = sample_rate
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")
    
    # Encode waveform as WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="WAV")
    buffer.seek(0)
    
    # Convert to base64 string
    return base64.b64encode(buffer.read()).decode("utf-8"), "WAV"

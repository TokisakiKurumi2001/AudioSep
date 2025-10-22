import os
import torch
import torchaudio
import numpy as np
from biodenoising import pretrained
from biodenoising.denoiser.dsp import convert_audio
from scipy.io import wavfile

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = pretrained.biodenoising16k_dns48().to(device)

AUDIO_PATH = "datasets/xcm_wav/XC458412 - Soundscape.wav"
wav, sr = torchaudio.load(os.path.join(AUDIO_PATH))
wav = convert_audio(wav, sr, model.sample_rate, model.chin).to(device)
with torch.no_grad():
    denoised = model(wav[None])[0]

denoise_sound = denoised.data.cpu().numpy()
rate = model.sample_rate
# Normalize and convert to int16 if needed
denoise_sound = denoise_sound.squeeze()  # Remove singleton dimensions
denoise_sound = denoise_sound / np.max(np.abs(denoise_sound))  # Normalize to [-1, 1]
denoise_sound = (denoise_sound * 32767).astype(np.int16)  # Convert to int16
wavfile.write(f"test_bio.wav", rate, denoise_sound)
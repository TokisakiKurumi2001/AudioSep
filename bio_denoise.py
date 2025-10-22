import os
import torch
import torchaudio
import numpy as np
from biodenoising import pretrained
from biodenoising.denoiser.dsp import convert_audio
from scipy.io import wavfile
import librosa
import shutil
from tqdm import tqdm

ORIG_PATH = 'data/org_data/birdsongs'
NEW_PATH = 'data/biodenoise'

class BioDenoise:
    def __init__(self):
        self.model, self.device = self.load_model()

    def load_model(self) -> "Tuple[model, device]":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = pretrained.biodenoising16k_dns48().to(device)
        return model, device

    def chunk(self, file: str, duration: int, tmp_path = "tmp"):
        shutil.rmtree(tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        # data, sample_rate = torchaudio.load(file)
        data, sample_rate = librosa.load(file, mono=True)

        total_len = int(len(data) / sample_rate)
        for offset in range(0, total_len, duration):
            # Calculate start and end samples
            start_sample = int(offset * sample_rate)
            end_sample = start_sample + int(duration * sample_rate)
            _data = data[start_sample:end_sample]
        
            output_path = os.path.join(tmp_path, f"sample_{offset}_{offset + duration}.wav")

            # Write the cut audio to a new wav file
            wavfile.write(output_path, sample_rate, _data)

        return tmp_path

    def __call__(self, wav_file) -> "Tuple[np.array, int]":
        folder_path = self.chunk(wav_file, duration = 600)
        ls = []
        rate = self.model.sample_rate
        for file in os.listdir(folder_path):
            _path = os.path.join(folder_path, file)

            wav, sr = torchaudio.load(_path)
            wav = convert_audio(wav, sr, self.model.sample_rate, self.model.chin).to(self.device)

            with torch.no_grad():
                denoised = self.model(wav[None])[0]

            denoise_sound = denoised.data.cpu().numpy()
            
            denoise_sound = denoise_sound.squeeze()
            denoise_sound = denoise_sound / np.max(np.abs(denoise_sound))
            denoise_sound = (denoise_sound * 32767).astype(np.int16)
            ls.append(denoise_sound)

        return np.concatenate(ls), rate

    def export(self, target_wav_file: str, denoise_sound, rate):
        wavfile.write(target_wav_file, rate, denoise_sound)

if __name__ == "__main__":
    denoiser = BioDenoise()
    os.makedirs(NEW_PATH, exist_ok=True)
    for folder in tqdm(os.listdir(ORIG_PATH)):
        new_folder_path = os.path.join(NEW_PATH, folder)
        os.makedirs(new_folder_path, exist_ok=True)
        folder_path = os.path.join(ORIG_PATH, folder)
        for wav_file in tqdm(os.listdir(folder_path)):
            denoise_sound, rate = denoiser(os.path.join(folder_path, wav_file))
            target_wav_file = os.path.join(new_folder_path, wav_file)
            denoiser.export(target_wav_file, denoise_sound, rate)

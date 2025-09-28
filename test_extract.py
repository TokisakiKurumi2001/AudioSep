from pipeline import build_audiosep, separate_audio
import os
import torch
from tqdm import tqdm
import librosa
from scipy.io import wavfile
import shutil
import numpy as np

ORG_PATH = "datasets/org_data/birdsongs"
OUTPUT_PATH = "datasets/audiosep"

def chunk(audio_file, duration: int, tmp_path: str = "tmp"):
    shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)
    data, sr = librosa.load(audio_file, mono=True)
    total_len = int(len(data) / sr)
    for offset in range(0, total_len, duration):
        start = int(offset * sr)
        end = start + int(duration * sr)
        _data = data[start:end]

        output_path = os.path.join(tmp_path, f"sample_{offset}_{offset + duration}.wav")
        wavfile.write(output_path, sr, _data)

    return tmp_path, total_len, duration

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_audiosep(
    config_yaml='config/audiosep_base.yaml', 
    checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt', 
    device=device
)

# loop through every .wav file in the dataset directory
for folder in tqdm(os.listdir(ORG_PATH)):
    folder_path = os.path.join(ORG_PATH, folder)
    output_folder_path = os.path.join(OUTPUT_PATH, folder)
    os.makedirs(output_folder_path, exist_ok=True)
    for file in tqdm(os.listdir(folder_path)):
        audio_file = os.path.join(folder_path, file)
        text = 'In the forest, the birds are singing.'
        chunk_path, total_len, duration = chunk(audio_file, duration = 100)
        ls = []
        out_file = os.path.join(output_folder_path, file)
        for idx in range(0, total_len, duration):
            chunk_audio_file = os.path.join(chunk_path, f"sample_{idx}_{idx + duration}.wav")
            separated_sound = separate_audio(model, chunk_audio_file, text, output_file="", device=device, should_write_to_file = False)
            ls.append(separated_sound)
        write_data = np.concatenate(ls)
        wavfile.write(out_file, 32_000, write_data)

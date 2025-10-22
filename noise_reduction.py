from scipy.io import wavfile
import noisereduce as nr
import os
from tqdm import tqdm
import librosa

# load data
DATASET_PATH = "datasets/xcm_wav"
OUTPUT_PATH = "nr_outputs"

def remove_noise(filename, output_dir, name):
    # rate, data = wavfile.read(filename)
    data, rate = librosa.load(filename, sr=None)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    print(f"Have reduced noise from {name}")
    wavfile.write(f"{output_dir}/no_noise_{name}.wav", rate, reduced_noise)

os.makedirs(OUTPUT_PATH, exist_ok=True)
for filename in tqdm(os.listdir(DATASET_PATH)):
    if filename.endswith('.mp3') or filename.endswith('.wav'):
        audio_file = os.path.join(DATASET_PATH, filename)
        remove_noise(audio_file, OUTPUT_PATH, filename)
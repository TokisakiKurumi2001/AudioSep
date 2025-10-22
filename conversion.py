from pydub import AudioSegment
import os
from tqdm import tqdm

datasets = "datasets/xcm"
output = "datasets/xcm_wav"
os.makedirs(output, exist_ok=True)
for filename in tqdm(os.listdir(datasets)):
    audio_file = os.path.join(datasets, filename)
    audio = AudioSegment.from_mp3(audio_file)
    new_filename = os.path.join(output, filename.split('.')[0] + '.wav')
    audio.export(new_filename, format="wav")
# You can then process the audio or convert it to a format compatible with scipy.io.wavfile
# For example, to export as WAV:
# audio.export("output.wav", format="wav")
import soundfile as sf
import os
import random
import pandas as pd
import librosa
import librosa.display
import numpy as np
from PIL import Image
from datasets import load_dataset

# Load the datasets
dataset1 = load_dataset("rjnieto/spanish-dialects", "argentina_female")['train']
dataset2 = load_dataset("rjnieto/spanish-dialects", "argentina_male")['train']

# Determine the number of samples to select (15% of the dataset size)
female_sample_size = int(len(dataset1) * 1)
male_sample_size = int(len(dataset2) * 1)

# Randomly sample 15% of the files
female_sampled = random.sample(list(dataset1), female_sample_size)
male_sampled = random.sample(list(dataset2), male_sample_size)

# Concatenate the sampled datasets
sampled_dataset = female_sampled + male_sampled

# Path to save files
stft_image_path = "/Users/nicolasadler/Documents/MUESemester3/MusicAndAI/Homeworks_Projects/Projects/Project_0/Project_0_2/stft_dialects_28_28"
os.makedirs(stft_image_path, exist_ok=True)

labels = []
# Function to compute and save STFT image
def save_stft_image(audio_data, sampling_rate, output_image_path):
    # Compute the STFT spectrogram
    stft = np.abs(librosa.stft(audio_data, n_fft=1024, hop_length=512))
    
    # Convert to decibels
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    
    # Normalize the spectrogram to range [0, 255]
    normalized_spectrogram = 255 * (stft_db - np.min(stft_db)) / (np.max(stft_db) - np.min(stft_db))
    normalized_spectrogram = normalized_spectrogram.astype(np.uint8)
    
    # Create a PIL Image from the spectrogram and resize it to 28x28
    image = Image.fromarray(normalized_spectrogram).convert('L')
    resized_image = image.resize((28, 28))
    
    # Save the image
    resized_image.save(output_image_path)

# Iterate over all examples in the sampled dataset
for idx, example in enumerate(sampled_dataset):
    # Access the audio
    audio_example = example['audio']
    audio_data = audio_example['array']
    sampling_rate = audio_example['sampling_rate']
    
    # Define output file names
    output_image_filename = os.path.join(stft_image_path, f'argentina_{idx}.png')
    
    # Save STFT image
    save_stft_image(audio_data, sampling_rate, output_image_filename)
    
    # Add the corresponding label
    if example['speaker_id'].startswith('female'):
        labels.append('argentina_female')
    else:
        labels.append('argentina_male')
    
    print(f"STFT image saved as '{output_image_filename}'")
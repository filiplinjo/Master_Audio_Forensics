import os
import numpy as np
import librosa
#import re
import shutil
import torch

SAMPLE_RATE = 48000


# Pad or trim each sample (impulse response needs to be truncated)
def pad_or_trim_spectrogram(spectrogram, target_size=(128, 4113)):
    """Pads or trims the spectrogram to the specified target size."""
    height_padding = target_size[0] - spectrogram.shape[0]
    width_padding = target_size[1] - spectrogram.shape[1]

    if height_padding > 0 or width_padding > 0:
        padding = ((max(0, height_padding), 0), (0, max(0, width_padding)))
        spectrogram = np.pad(spectrogram, pad_width=padding, mode='constant')
    else:
        spectrogram = spectrogram[:target_size[0], :target_size[1]]
    return spectrogram

def wav_to_spectrogram(wav_path, sr=SAMPLE_RATE, n_fft=2048, hop_length=128,
                       n_mels=128, max_length=4113, use_log=True):
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    if use_log:
        # Compute Log Mel Spectrogram
        spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    else:
        # Use Mel Spectrogram directly
        spectrogram = mel_spectrogram

    # Pad or trim to the target size
    spectrogram = pad_or_trim_spectrogram(spectrogram, target_size=(n_mels, max_length))



# class PCENTransform(object):
#     def __init__(self, sr=48000, hop_length=256, gain=0.9, bias=20, power=0.25, time_constant=0.08, eps=1e-6):
#         self.sr = sr
#         self.hop_length = hop_length
#         self.gain = gain
#         self.bias = bias
#         self.power = power
#         self.time_constant = time_constant
#         self.eps = eps

#     def __call__(self, audio):
#         if isinstance(audio, torch.Tensor):
#             audio = audio.numpy()  # Convert to NumPy for librosa processing

#         audio = np.clip(audio, a_min=self.eps, a_max=None)
#         pcen_audio = librosa.pcen(audio * (2 ** 31), sr=self.sr, hop_length=self.hop_length,
#                                   gain=self.gain, bias=self.bias, power=self.power,
#                                   time_constant=self.time_constant, eps=self.eps)

#         return torch.from_numpy(pcen_audio).float()  # Convert back to tensor

# def compute_global_stats(input_directory, sr=48000, n_fft=2048, hop_length=512, n_mels=128):
#     log_mel_values = []

#     for filename in sorted(os.listdir(input_directory)):
#         if filename.endswith('.wav'):
#             filepath = os.path.join(input_directory, filename)
#             try:
#                 y, sr = librosa.load(filepath, sr=sr)
#                 mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#                 log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
#                 log_mel_values.append(log_mel_spectrogram.flatten())
#             except Exception as e:
#                 print(f"An error occurred while processing {filename}: {e}")

#     # Concatenate all values and compute global mean and standard deviation
#     all_values = np.concatenate(log_mel_values, axis=0)
#     global_mean = np.mean(all_values)
#     global_std = np.std(all_values)

#     return global_mean, global_std

def copy_mix_files(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if 'mix' in file and file.endswith('.npy'):  # Ensure only .npy files are copied
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(destination_directory, file)

                shutil.copy2(src_file_path, dest_file_path)
                #print(f"Copied mix file {file} to {dest_file_path}")

# def extract_background_noise(wav_path, pcen_transform):
#     try:
#         y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
#         mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

#         # Apply PCEN transform
#         pcen_spectrogram = pcen_transform(mel_spectrogram)

#         # Subtract PCEN-transformed signal from original
#         background_spectrogram = mel_spectrogram - pcen_spectrogram

#         return background_spectrogram
#     except Exception as e:
#         print(f"An error occurred while processing {wav_path}: {e}")
#         return None


def split_directory(input_dir, output_dir1, output_dir2):
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # Ensure output directories exist
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # List all items (files and directories)
    items = sorted(os.listdir(input_dir))
    total_items = len(items)
    
    # Calculate the number of items to put in each output directory
    items_per_directory = total_items // 2

    # Copy the first 15 items to output_dir1
    for item in items[:items_per_directory]:
        src = os.path.join(input_dir, item)
        dest = os.path.join(output_dir1, item)
        if os.path.isdir(src):
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)

    # Copy the next 15 items to output_dir2
    for item in items[items_per_directory:items_per_directory*2]:
        src = os.path.join(input_dir, item)
        dest = os.path.join(output_dir2, item)
        if os.path.isdir(src):
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)

    print(f"Finished copying from {input_dir}")


def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in sorted(os.listdir(input_directory)):
        if filename.endswith('.wav'):
            src_path = os.path.join(input_directory, filename)
            prefix = filename.split('_')[0]

            # Usage example
            mel_spectrogram = wav_to_spectrogram(src_path, use_log=False) # For Mel Spectrogram
            mel_spectrogram = wav_to_spectrogram(src_path, use_log=True) # For Log-Mel Spectrogram

            if mel_spectrogram is None:
                continue

            npy_filename = f"{os.path.splitext(filename)[0]}.npy"
            entity_type = "a" if "_a_" in filename else "b"
            entity_dir = os.path.join(output_directory, f"{prefix}_{entity_type}_entity")
            os.makedirs(entity_dir, exist_ok=True)
            npy_path = os.path.join(entity_dir, npy_filename)
            np.save(npy_path, mel_spectrogram)


if __name__ == "__main__":
    input_directory = "original_training"
    output_directory = "training_output_final_Dataset"
    main(input_directory, output_directory)

    input_directory_testing = "original_testing"
    output_directory_testing_loss = "output1"
    output_directory_testing_evaluation = "output2"
    split_directory(input_directory_testing, output_directory_testing_loss, output_directory_testing_evaluation)

    main(output_directory_testing_loss, "testing_output_final_Dataset_loss")
    main(output_directory_testing_evaluation, "testing_output_final_Dataset_evaluation")

    if os.path.exists("output1"):
        shutil.rmtree("output1")
    if os.path.exists("output2"):
        shutil.rmtree("output2")

    # Copy only 'mix' files from output_directory_testing_evaluation to a new directory
    new_directory_for_mix_files = "evaluation_with_only_mix"
    copy_mix_files("testing_output_final_Dataset_evaluation", new_directory_for_mix_files)
    print("All Done")



    #Now we need keep both output_directory_testing_loss and output_directory_testing_evaluation as they are, but also make a new folder
    #That should contain output_directory_testing_evaluation where we only have the files that contains "mix"

# This code is used to generate the datasets where we want to have; impulse response, revberant audio, and the mix audio
# There is 5 samples per class in the original_traning dataset, where these are seperated into two entities of the class
# Each entity have their own unique reverberant audio, and mix, however, they share the same impulse response
# This is because they in practice are two different audio recordings in the same room enviroment.

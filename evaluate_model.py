import os
import torch
import numpy as np
from torchvision import transforms
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import librosa
from sklearn.metrics.pairwise import cosine_similarity

# class GaussianNoise(object):
#     def __init__(self, min_std=0.01, max_std=0.2):
#         self.min_std = min_std
#         self.max_std = max_std

#     def add_gaussian_noise(self, audio):
#         # Randomly choose a standard deviation value within the specified range
#         std_dev = np.random.uniform(self.min_std, self.max_std)
#         noise = np.random.normal(0, std_dev, audio.shape)
#         return audio + noise

#     def __call__(self, audio):
#         return torch.from_numpy(self.add_gaussian_noise(audio.numpy())).float()


# class PinkNoise:
#     def __init__(self, min_std=0.03, max_std=0.3):
#         self.min_std = min_std
#         self.max_std = max_std

#     def add_pink_noise(self, audio):
#         # Dynamically choose a standard deviation value
#         std_dev = np.random.uniform(self.min_std, self.max_std)
#         # Generate white noise with this standard deviation
#         white_noise = np.random.normal(0, std_dev, audio.shape)
#         # Apply a filter to convert white noise to pink noise
#         fft = np.fft.rfft(white_noise)
#         pink_noise = np.fft.irfft(fft / np.sqrt(np.arange(1, len(fft) + 1)), n=audio.shape[-1])
#         return audio + pink_noise

#     def __call__(self, audio):
#         return torch.from_numpy(self.add_pink_noise(audio.numpy())).float()

# class PCENTransform(object):
#     def __init__(self, sr=48000, hop_length=256, gain=0.8, bias=10, power=0.35, time_constant=0.06, eps=1e-6):
#         self.sr = sr
#         self.hop_length = hop_length
#         self.gain = gain
#         self.bias = bias
#         self.power = power
#         self.time_constant = time_constant
#         self.eps = eps

#     def __call__(self, audio):
#         # Ensure there are no zero values before taking the log
#         audio = np.clip(audio, a_min=self.eps, a_max=None)

#         # Apply PCEN
#         pcen_audio = librosa.pcen(audio * (2 ** 31), sr=self.sr, hop_length=self.hop_length,
#                                    gain=self.gain, bias=self.bias, power=self.power,
#                                    time_constant=self.time_constant, eps=self.eps)

#         return pcen_audio  # Returning as a NumPy array


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Convolutional layers with max pooling
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 5), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        # Calculate the size of the flattened feature maps (_get_flat_feature_size(self))
        self.flat_feature_size = self._get_flat_feature_size()
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flat_feature_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def _get_flat_feature_size(self):
        # Set the input shape directly here
        dummy_input = torch.zeros(1, 1, 128, 4126)  # Updated size
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        return int(torch.prod(torch.tensor(dummy_output.size()[1:])))


    # Forward pass for one side of the Siamese net
    def forward_once(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Forward pass for both sides of the Siamese net
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# transform = transforms.Compose([
#     GaussianNoise(),
#     PinkNoise(),
#     PCENTransform()
# ])

def find_correct_match_index(file, sample_paths):
    base_name = os.path.splitext(os.path.basename(file))[0]
    main_part = base_name.split('_')[0]
    for index, path in enumerate(sample_paths):
        other_base_name = os.path.splitext(os.path.basename(path))[0]
        other_main_part = other_base_name.split('_')[0]
        if main_part == other_main_part and file != path:
            return index
    return -1

def is_correct_match(file1, file2):
    # Extract the numeric part of the filename before '_'
    base_name1 = os.path.splitext(os.path.basename(file1))[0]
    base_name2 = os.path.splitext(os.path.basename(file2))[0]
    numeric_part1 = base_name1.split('_')[0]
    numeric_part2 = base_name2.split('_')[0]
    return numeric_part1 == numeric_part2


def compare_audio_samples(sample_folder):
    # Get a list of all sample file paths in the folder
    sample_files = os.listdir(sample_folder)
    sample_paths = [os.path.join(sample_folder, file) for file in sample_files]

    embeddings = []

    # Load, preprocess and add batch and channel dimensions for all samples
    for sample_path in sample_paths:
        sample = np.load(sample_path)
        sample_tensor = torch.from_numpy(sample).float()  # Convert to tensor and ensure float32
        sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        with torch.no_grad():
            embedding = model.forward_once(sample_tensor)
            embeddings.append(embedding)

    num_samples = len(sample_paths)
    distance_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                # Calculate Euclidean distance
                distance = torch.norm(embeddings[i] - embeddings[j]).cpu().numpy()
                distance_matrix[i, j] = distance

    closest_samples = []
    closest_distances = []
    most_distant_samples = []
    most_distant_distances = []
    total_distances = []
    correct_match_distances = []
    correct_matches = 0

    for i in range(len(sample_paths)):
        # Find the closest and most distant samples excluding the sample itself
        closest_index = np.argmin([dist if idx != i else np.inf for idx, dist in enumerate(distance_matrix[i])])
        most_distant_index = np.argmax([dist if idx != i else -np.inf for idx, dist in enumerate(distance_matrix[i])])
        
        closest_sample = sample_paths[closest_index]
        most_distant_sample = sample_paths[most_distant_index]
        closest_distance = distance_matrix[i, closest_index]
        most_distant_distance = distance_matrix[i, most_distant_index]

        closest_samples.append(closest_sample)
        closest_distances.append(closest_distance)
        most_distant_samples.append(most_distant_sample)
        most_distant_distances.append(most_distant_distance)

        # Calculate the total distance to all other samples
        total_distances.append(np.sum([dist for idx, dist in enumerate(distance_matrix[i]) if idx != i]))

        correct_match_index = find_correct_match_index(sample_paths[i], sample_paths)
        correct_match_distance = distance_matrix[i, correct_match_index] if correct_match_index != -1 else None
        correct_match_distances.append(correct_match_distance)

        correct_match = correct_match_index == closest_index
        correct_matches += int(correct_match)

    # Calculate average distance
    average_distances = [total / (len(sample_paths) - 1) for total in total_distances]
    
    accuracy = (correct_matches / len(sample_paths)) * 100
    return sample_paths, closest_samples, closest_distances, most_distant_samples, most_distant_distances, average_distances, correct_match_distances, correct_matches, accuracy


# Initialize the model
model = SiameseNetwork()

# Load the model from a saved state
model_path = 'best_siamese_model_dual_11.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Main part of the script
if __name__ == '__main__':
    sample_folder = 'evaluation_with_only_mix/'
    sample_paths, closest_samples, closest_distances, most_distant_samples, most_distant_distances, average_distances, correct_match_distances, correct_matches, accuracy = compare_audio_samples(sample_folder)

    for i, sample_path in enumerate(sample_paths):
        correct_match = is_correct_match(sample_path, closest_samples[i])
        star = '*' if correct_match else ' '
        correct_match_distance = correct_match_distances[i]
        print(f"{star}Sample {os.path.basename(sample_path)} is closest to Sample {os.path.basename(closest_samples[i])} with a distance of {closest_distances[i]:.4f}")
        if correct_match_distance is not None:
            print(f"   Correct match distance: {correct_match_distance:.4f}")
        print(f"   Most distant sample is Sample {os.path.basename(most_distant_samples[i])} with a distance of {most_distant_distances[i]:.4f}")
        print(f"   Average distance to other samples: {average_distances[i]:.4f}")

    print(f"Number of correct matches: {correct_matches}/{len(sample_paths)}")
    print(f"Accuracy score: {accuracy:.2f}%")

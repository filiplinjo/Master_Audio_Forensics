import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F
import librosa

random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

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
#         if isinstance(audio, torch.Tensor):
#             audio = audio.numpy()  # Convert to NumPy for librosa processing

#         audio = np.clip(audio, a_min=self.eps, a_max=None)
#         pcen_audio = librosa.pcen(audio * (2 ** 31), sr=self.sr, hop_length=self.hop_length,
#                                   gain=self.gain, bias=self.bias, power=self.power,
#                                   time_constant=self.time_constant, eps=self.eps)

#         return torch.from_numpy(pcen_audio).float()  # Convert back to tensor

#15 pairs positve
# def prepare_data(directory):
#     triplets = []

#     # First, construct a global list of all samples.
#     all_samples = {}
#     class_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
#     class_folders.sort()

#     for folder_name in class_folders:
#         folder_path = os.path.join(directory, folder_name)
#         all_samples[folder_name] = [os.path.join(folder_path, s) for s in os.listdir(folder_path)]

#     for class_name, class_samples in all_samples.items():
#         # Generate all unique pairs within the class
#         for i in range(len(class_samples)):
#             for j in range(i + 1, len(class_samples)):
#                 anchor_sample_path = class_samples[i]
#                 positive_sample_path = class_samples[j]

#                 # Select a random negative class different from the current class
#                 negative_classes = [name for name in class_folders if name != class_name]
#                 negative_class = random.choice(negative_classes)
#                 negative_sample_path = random.choice(all_samples[negative_class])

#                 triplets.append((anchor_sample_path, positive_sample_path, negative_sample_path))

#     return triplets

def prepare_initial_triplets(directory):
    initial_triplets = []

    # Dynamically determine class IDs based on folder names
    class_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    class_ids = set(folder.split('_')[0] for folder in class_folders)

    for class_id in class_ids:
        # Paths for a_entity and b_entity
        a_entity_path = os.path.join(directory, f"{class_id}_a_entity")
        b_entity_path = os.path.join(directory, f"{class_id}_b_entity")

        # Paths for mix, reverb, and impulse response within a_entity and b_entity
        a_mix = os.path.join(a_entity_path, f"{class_id}_a_mix.npy")
        a_reverb = os.path.join(a_entity_path, f"{class_id}_a_reverb.npy")
        a_impulse = os.path.join(a_entity_path, f"{class_id}_impulse_response.npy")
        b_mix = os.path.join(b_entity_path, f"{class_id}_b_mix.npy")
        b_reverb = os.path.join(b_entity_path, f"{class_id}_b_reverb.npy")
        b_impulse = os.path.join(b_entity_path, f"{class_id}_impulse_response.npy")

        # Generate triplets for a_entity
        if os.path.isfile(a_mix) and os.path.isfile(a_reverb):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((a_mix, a_reverb, negative_sample))
        if os.path.isfile(a_mix) and os.path.isfile(a_impulse):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((a_mix, a_impulse, negative_sample))
        if os.path.isfile(a_mix) and os.path.isfile(b_mix):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((a_mix, b_mix, negative_sample))
        if os.path.isfile(a_mix) and os.path.isfile(b_reverb):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((a_mix, b_reverb, negative_sample))

        # Generate triplets for b_entity
        if os.path.isfile(b_mix) and os.path.isfile(b_reverb):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((b_mix, b_reverb, negative_sample))
        if os.path.isfile(b_mix) and os.path.isfile(b_impulse):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((b_mix, b_impulse, negative_sample))
        if os.path.isfile(b_mix) and os.path.isfile(a_reverb):
            negative_sample = get_random_negative_sample(directory, class_id)
            initial_triplets.append((b_mix, a_reverb, negative_sample))

    #print("All triplets are generated: ", len(initial_triplets))
    #print(len(initial_triplets))
    return initial_triplets




def get_random_negative_sample(directory, current_class_id):
    # List all class folders
    class_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(current_class_id) == False]

    #if not class_folders:
    #    raise ValueError("No other class folders found")

    # Choose a random class folder
    negative_class_folder = random.choice(class_folders)
    negative_class_path = os.path.join(directory, negative_class_folder)

    # Choose a random negative sample from the chosen folder
    negative_samples = [file for file in os.listdir(negative_class_path) if file.endswith(".npy")]
    #if not negative_samples:
    #    raise ValueError(f"No negative samples found in {negative_class_path}")

    negative_sample = random.choice(negative_samples)
    #print(negative_sample)
    return os.path.join(negative_class_path, negative_sample)


def update_triplets_for_epoch(initial_triplets, all_samples, epoch):
    updated_triplets = []
    random.seed(epoch)  # Seed for reproducibility

    for anchor, positive, _ in initial_triplets:
        # Randomly select a different class
        random_class_id = random.choice(list(all_samples.keys()))
        while random_class_id == anchor.split("/")[-2]:  # Ensure it's a different class
            random_class_id = random.choice(list(all_samples.keys()))

        # Randomly select a sample from the random class as the negative sample
        negative_sample = random.choice(all_samples[random_class_id])

        # Create updated triplet (anchor, positive, negative)
        updated_triplets.append((anchor, positive, negative_sample))

    #print(len( updated_triplets))
    return updated_triplets



class GaussianNoise(object):
    def __init__(self, min_std=0.001, max_std=0.3):
        self.min_std = min_std
        self.max_std = max_std

    def add_gaussian_noise(self, audio):
        # Randomly choose a standard deviation value within the specified range
        std_dev = np.random.uniform(self.min_std, self.max_std)
        noise = np.random.normal(0, std_dev, audio.shape)
        return audio + noise

    def __call__(self, audio):
        return torch.from_numpy(self.add_gaussian_noise(audio.numpy())).float()

#try increase max noise

class PinkNoise:
    def __init__(self, min_std=0.003, max_std=0.4):
        self.min_std = min_std
        self.max_std = max_std

    def add_pink_noise(self, audio):
        # Dynamically choose a standard deviation value between threshold
        std_dev = np.random.uniform(self.min_std, self.max_std)
        # Generate white noise with this standard deviation
        white_noise = np.random.normal(0, std_dev, audio.shape)
        # Apply a filter to convert white noise to pink noise
        fft = np.fft.rfft(white_noise)
        pink_noise = np.fft.irfft(fft / np.sqrt(np.arange(1, len(fft) + 1)), n=audio.shape[-1])
        return audio + pink_noise

    def __call__(self, audio):
        return torch.from_numpy(self.add_pink_noise(audio.numpy())).float()

class TripletDataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform
        self.data = {}

        # Pre-load data into memory
        for anchor_path, positive_path, negative_path in self.triplets:
            if anchor_path not in self.data:
                self.data[anchor_path] = np.load(anchor_path)
            if positive_path not in self.data:
                self.data[positive_path] = np.load(positive_path)
            if negative_path not in self.data:
                self.data[negative_path] = np.load(negative_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor = torch.tensor(self.data[anchor_path], dtype=torch.float32).unsqueeze(0)
        positive = torch.tensor(self.data[positive_path], dtype=torch.float32).unsqueeze(0)
        negative = torch.tensor(self.data[negative_path], dtype=torch.float32).unsqueeze(0)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

def test_model(model, criterion, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            outputs_anchor = model.forward_once(anchor)
            outputs_positive = model.forward_once(positive)
            outputs_negative = model.forward_once(negative)

            loss = criterion(outputs_anchor, outputs_positive, outputs_negative)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


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
        dummy_input = torch.zeros(1, 1, 128, 4113)  # Updated size
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


# Med L2 normalization, test:
# def forward_one(self, x):
#     x = self.conv(x)
#        #print("Shape after conv layers:", x.shape)
#        #x = self.global_avg_pool(x)  # Apply global average pooling
#     x = x.view(x.size(0), -1)  # Flatten output
#     x = self.fc(x)
#     x = F.normalize(x, p=2, dim=1)  # L2 normalization
#     return x

transform = transforms.Compose([
    GaussianNoise(),
    PinkNoise(),
#    PCENTransform(),
])

# def compute_conv_output_shape():
#     YOUR_HEIGHT = 13
#     YOUR_WIDTH = 1033
#     dummy_input = torch.randn(1, 1, YOUR_HEIGHT, YOUR_WIDTH)
#     output = siamese_net.conv(dummy_input)
#     print(output.size())

learning_rate = 0.0002
num_epochs = 100
margin = 0.5

siamese_net = SiameseNetwork()
criterion = TripletLoss(margin)
optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate, weight_decay=1e-4)

best_loss = float('inf')
patience = 7
early_stopping_counter = 0
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience)

def main():
    train_directory = 'training_output_final_Dataset'
    test_directory = 'testing_output_final_Dataset_loss'

    # Generate initial triplets for training and testing datasets
    train_triplets = prepare_initial_triplets(train_directory)
    test_triplets = prepare_initial_triplets(test_directory)

    # Initialize the all_samples dictionary to store sample information
    all_samples = {}  # This line initializes the all_samples dictionary

    # Populate the all_samples dictionary with paths to samples, organized by class
    class_folders = [f for f in os.listdir(train_directory) if os.path.isdir(os.path.join(train_directory, f))]
    for class_folder in class_folders:
        sample_paths = [os.path.join(train_directory, class_folder, f) 
                        for f in os.listdir(os.path.join(train_directory, class_folder)) 
                        if f.endswith('.npy')]
        all_samples[class_folder] = sample_paths

    # Initialize the datasets and loaders
    #train_dataset = TripletDataset(train_triplets, transform=transform)
    test_dataset = TripletDataset(test_triplets, transform=transform)
    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize the Siamese Network and other components
    siamese_net = SiameseNetwork().to(device)
    criterion = TripletLoss(margin)
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience)

    global best_loss, early_stopping_counter
    early_stopping_counter = 0
    best_loss = float('inf')
    print("Training Progress:")

    for epoch in range(num_epochs):
        siamese_net.train()
        total_loss = 0.0

        # Update training triplets for the new epoch
        train_triplets = update_triplets_for_epoch(train_triplets, all_samples, epoch)
        train_dataset = TripletDataset(train_triplets, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            # Convert input tensors to float32 and transfer to the appropriate device
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            # Process each image through the network
            outputs_anchor = siamese_net.forward_once(anchor)
            outputs_positive = siamese_net.forward_once(positive)
            outputs_negative = siamese_net.forward_once(negative)

            # Calculate loss
            loss = criterion(outputs_anchor, outputs_positive, outputs_negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Print training progress
            progress = (batch_idx + 1) * 100 / len(train_loader)
            if progress % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Progress: {progress:.0f}%')

        avg_loss = total_loss / len(train_loader)
        test_loss = test_model(siamese_net, criterion, test_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}')
        scheduler.step(test_loss)

        # Check for best loss and early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            early_stopping_counter = 0
            torch.save(siamese_net.state_dict(), 'best_siamese_model_6.pth')
        else:
            early_stopping_counter += 1

        if early_stopping_counter > patience:
            print("Early stopping!")
            break

    print("Finished Training")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")



#Test both mfcc and melspectogram concat

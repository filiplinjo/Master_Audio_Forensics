import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import random
import librosa

random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def prepare_data(directory, ratio_positive_negative=1):
    entity_folders = sorted(os.listdir(directory))
    folder_to_base = {folder: folder.split('_')[0] for folder in entity_folders if os.path.isdir(os.path.join(directory, folder))}
    base_to_folders = {base: [folder for folder in folder_to_base if folder_to_base[folder] == base] for base in set(folder_to_base.values())}

    pairs = []
    labels = []

    for base, folders in base_to_folders.items():
        if len(folders) != 2:
            continue

        a_entity, b_entity = sorted(folders)
        sample1 = os.path.join(directory, a_entity, a_entity.split('_entity')[0] + '_mix.npy')
        sample2 = os.path.join(directory, b_entity, b_entity.split('_entity')[0] + '_mix.npy')
        pairs.append((sample1, sample2))
        labels.append(1)

        other_bases = {other_base: folders for other_base, folders in base_to_folders.items() if other_base != base and len(folders) == 2}
        if len(other_bases) < ratio_positive_negative:
            continue

        selected_neg_bases = random.sample(list(other_bases.keys()), ratio_positive_negative)
        for neg_base in selected_neg_bases:
            neg_folder = random.choice(other_bases[neg_base])
            sample2_neg = os.path.join(directory, neg_folder, neg_folder.split('_entity')[0] + '_mix.npy')
            pairs.append((sample1, sample2_neg))
            labels.append(0)
    #print(pairs)
    return pairs, labels

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

class PairwiseDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_path, positive_path = self.pairs[idx]
        label = self.labels[idx]

        # Load data directly from the .npy files
        anchor = torch.tensor(np.load(anchor_path), dtype=torch.float32).unsqueeze(0)
        positive = torch.tensor(np.load(positive_path), dtype=torch.float32).unsqueeze(0)
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
        #print(f"Returning from __getitem__: {anchor.shape}, {positive.shape}, {label}")
        return anchor, positive, torch.tensor(label, dtype=torch.float32)


def test_model_phase2(model, criterion, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for anchor, positive, label in test_loader:
            anchor, positive = anchor.to(device), positive.to(device)
            label = label.to(device)

            output1, output2 = model(anchor, positive)
            loss = criterion(output1, output2, label)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


class GaussianNoise(object):
    def __init__(self, min_std=0, max_std=0.2):
        self.min_std = min_std
        self.max_std = max_std

    def add_gaussian_noise(self, audio):
        # Randomly choose a standard deviation value within the specified range
        std_dev = np.random.uniform(self.min_std, self.max_std)
        noise = np.random.normal(0, std_dev, audio.shape)
        return audio + noise

    def __call__(self, audio):
        return torch.from_numpy(self.add_gaussian_noise(audio.numpy())).float()


class PinkNoise:
    def __init__(self, min_std=0, max_std=0.3):
        self.min_std = min_std
        self.max_std = max_std

    def add_pink_noise(self, audio):
        # Dynamically choose a standard deviation value
        std_dev = np.random.uniform(self.min_std, self.max_std)
        # Generate white noise with this standard deviation
        white_noise = np.random.normal(0, std_dev, audio.shape)
        # Apply a filter to convert white noise to pink noise
        fft = np.fft.rfft(white_noise)
        pink_noise = np.fft.irfft(fft / np.sqrt(np.arange(1, len(fft) + 1)), n=audio.shape[-1])
        return audio + pink_noise

    def __call__(self, audio):
        return torch.from_numpy(self.add_pink_noise(audio.numpy())).float()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Normalize the outputs to have unit length
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        # Contrastive loss
        loss_contrastive = (1 - label) * torch.pow(euclidean_distance, 2) + \
                           label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return loss_contrastive.mean()

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
        dummy_input = torch.zeros(1, 1, 128, 4119)  # Updated size
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

# print("Shape after conv layers:", x.shape)
# x = self.global_avg_pool(x)  # Apply global average pooling

def update_pairs_for_epoch(pairs, directory, epoch):
    updated_pairs = []
    random.seed(epoch)  # Seed for reproducibility

    for anchor, _ in pairs:
        anchor_class = anchor.split("/")[-2].split('_')[0]

        while True:
            # Randomly select a folder and a file within that folder
            random_folder = random.choice(os.listdir(directory))
            random_file = random.choice(os.listdir(os.path.join(directory, random_folder)))
            negative_sample = os.path.join(directory, random_folder, random_file)

            # Check if the negative sample is from a different class
            if anchor_class not in negative_sample:
                break

        updated_pairs.append((anchor, negative_sample))

    return updated_pairs


transform = transforms.Compose([
     GaussianNoise(),
     PinkNoise(),
 #    PCENTransform(),
 ])

learning_rate = 0.0001
num_epochs = 100
margin = 1.0
patience = 7

def main():
    # Data Preparation
    train_pairs, train_labels = prepare_data('training_output_final_Dataset')
    test_pairs, test_labels = prepare_data('testing_output_final_Dataset_loss')

    # Network and Optimizer setup
    siamese_net = SiameseNetwork().to(device)
    pretrained_weights = torch.load('best_siamese_model_5.pth')
    siamese_net.load_state_dict(pretrained_weights, strict=True)
    contrastive_loss = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience)

    # DataLoader setup
    test_dataset = PairwiseDataset(test_pairs, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    # Training loop
    best_loss = float('inf')
    early_stopping_counter = 0
    for epoch in range(100):
        updated_train_pairs = update_pairs_for_epoch(train_pairs, 'training_output_final_Dataset', epoch)
        updated_train_labels = [1] * len(train_pairs) + [0] * (len(updated_train_pairs) - len(train_pairs))
        train_dataset = PairwiseDataset(updated_train_pairs, updated_train_labels, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

        siamese_net.train()
        total_loss = 0.0
        for sample1_batch, sample2_batch, label_batch in train_loader:
            sample1_batch = sample1_batch.to(device).float()
            sample2_batch = sample2_batch.to(device).float()
            label_batch = label_batch.to(device).float()
            optimizer.zero_grad()
            output1_batch, output2_batch = siamese_net(sample1_batch, sample2_batch)
            loss = contrastive_loss(output1_batch, output2_batch, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_test_loss = test_model_phase2(siamese_net, contrastive_loss, test_loader)
        print(f'Epoch [{epoch + 1}/100], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            early_stopping_counter = 0
            torch.save(siamese_net.state_dict(), 'best_siamese_model_dual_11.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 4:
                print("Early stopping triggered.")
                break

        scheduler.step(avg_test_loss)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")

# Log-mel spectrograms i stedet for pcen? eller concat begge?




#fiks validation set på samme måte som negative endringer
#endre i slutten av hver epoch for begge


#Optmizer, or kernel

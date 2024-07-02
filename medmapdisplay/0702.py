import os
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib

# OpenMP error avoidance
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Data path setting
example_person_dicom_dir_path = "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0010_20190710_2"

# Normalization function
def normalize_ct_image(image):
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

# Denormalize function
def denormalize_ct_image(image):
    scaler = MinMaxScaler()
    return scaler.inverse_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

# Data preparation
def load_and_preprocess_dicom_files(dicom_dir):
    CTfiles = []
    dicomfilepaths = sorted(os.listdir(dicom_dir))
    for filepath in dicomfilepaths:
        full_path = os.path.join(dicom_dir, filepath)
        d = pydicom.dcmread(full_path)
        hu = d.pixel_array
        nhu = normalize_ct_image(hu)
        CTfiles.append(nhu)
    CTfiles = np.array(CTfiles)
    return CTfiles

# Get normalized data
mizumashi_of_normalizedCTfiles = load_and_preprocess_dicom_files(example_person_dicom_dir_path)

# Ensure the data is 3D
if mizumashi_of_normalizedCTfiles.ndim == 3:
    mizumashi_of_normalizedCTfiles = np.expand_dims(mizumashi_of_normalizedCTfiles, axis=0)

# Patch extraction
def extract_3d_patches(scan, patch_size, num_patches):
    patches = []
    if scan.ndim == 3:
        x_max, y_max, z_max = scan.shape
    else:
        raise ValueError("Scan does not have 3 dimensions")
    
    for _ in range(num_patches):
        x = np.random.randint(0, x_max - patch_size[0])
        y = np.random.randint(0, y_max - patch_size[1])
        z = np.random.randint(0, z_max - patch_size[2])
        patch = scan[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]
        patches.append(patch)
    return np.array(patches)

# Extract 3D patches
patch_size = (32, 32, 32)
num_patches_per_scan = 200
all_patches = []
for scan in mizumashi_of_normalizedCTfiles:
    patches = extract_3d_patches(scan, patch_size, num_patches_per_scan)
    all_patches.extend(patches)

# Convert patches to tensor
all_patches = np.array(all_patches)
tensor_patches = torch.tensor(all_patches, dtype=torch.float32).unsqueeze(1)

# Hyperparameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
Z_DIM = 100
EPOCHS = 20

# 3D Encoder definition
class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 8 * 8 * 8, Z_DIM)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 3D Decoder definition
class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        self.fc = nn.Linear(Z_DIM, 128 * 8 * 8 * 8)
        self.conv1 = nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose3d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

# DataLoader creation
dataset = TensorDataset(tensor_patches)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model definition
encoder = Encoder3D()
decoder = Decoder3D()

# Optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# Loss function
criterion = nn.MSELoss()

# Training loop
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    total_train_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        z = encoder(x)
        x_hat = decoder(z)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(dataloader)
    train_losses.append(avg_train_loss)
    
    if (epoch + 1) % 5 == 0:
        encoder.eval()
        decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                z = encoder(x)
                x_hat = decoder(z)
                val_loss = criterion(x_hat, x)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(dataloader)
        val_losses.append(avg_val_loss)
        print(f'Epoch: {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        encoder.train()
        decoder.train()

# Reconstruct images
z_points = encoder(tensor_patches)
reconst_images = decoder(z_points)

# Plot loss curve
def plot_loss_curve(train_losses, val_losses, save_path="results"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.show()

# Plot images
def plot_images(original, decoded, n=5, save_path="results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure(figsize=(15, 10))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i, 0, :, :, 16], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        denormalized_image = denormalize_ct_image(decoded[i, 0, :, :, 16].detach().numpy())
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(denormalized_image, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        difference = denormalized_image - original[i, 0, :, :, 16]
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(difference, cmap="bwr", vmin=-1, vmax=1)
        plt.title("Difference")
        plt.axis("off")

    plt.savefig(os.path.join(save_path, "reconstructed_images.png"))
    plt.show()

# Plot and save loss curve
plot_loss_curve(train_losses, val_losses)

# Plot and save reconstructed images
plot_images(tensor_patches, reconst_images.detach(), n=5)

print("Training and visualization complete.")

# Save models
model_save_path = "saved_models"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

torch.save(encoder.state_dict(), os.path.join(model_save_path, "encoder.pth"))
torch.save(decoder.state_dict(), os.path.join(model_save_path, "decoder.pth"))

print("Models saved successfully.")

# Load models for transfer learning
def load_models(encoder_path, decoder_path):
    encoder = Encoder3D()
    decoder = Decoder3D()
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder

encoder, decoder = load_models(os.path.join(model_save_path, "encoder.pth"), os.path.join(model_save_path, "decoder.pth"))

print("Models loaded successfully for transfer learning.")

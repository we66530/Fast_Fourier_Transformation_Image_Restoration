import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import time
from tqdm import tqdm
import torch.nn.functional as F

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, masked_dir, unmasked_dir, transform=None):
        self.masked_paths = [os.path.join(masked_dir, f) for f in os.listdir(masked_dir) if f.endswith('.jpg')]
        self.unmasked_paths = [os.path.join(unmasked_dir, f) for f in os.listdir(unmasked_dir) if f.endswith('.jpg')]
        self.transform = transform

        assert len(self.masked_paths) == len(self.unmasked_paths), \
            "The number of masked and unmasked images must be the same."

    def __len__(self):
        return len(self.masked_paths)

    def __getitem__(self, idx):
        masked_image = Image.open(self.masked_paths[idx]).convert("L")
        unmasked_image = Image.open(self.unmasked_paths[idx]).convert("L")

        if self.transform:
            masked_image = self.transform(masked_image)
            unmasked_image = self.transform(unmasked_image)

        masked_fft = torch.fft.fft2(masked_image)
        unmasked_fft = torch.fft.fft2(unmasked_image)

        masked_fft = torch.stack([masked_fft.real, masked_fft.imag], dim=0)
        unmasked_fft = torch.stack([unmasked_fft.real, unmasked_fft.imag], dim=0)

        return masked_fft, unmasked_fft


# Neural Network
class FFTRestorationNet(nn.Module):
    def __init__(self):
        super(FFTRestorationNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# SSIM Loss Function (using pytorch)
def ssim_loss(x, y, max_val=1.0):
    _, channel, height, width = x.size()

    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))

    return torch.mean(1 - ssim_map)


# Combined Loss Function
def combined_loss(output, target, alpha=0.8):
    l1 = nn.L1Loss()(output, target)
    ssim = ssim_loss(output, target)
    return alpha * l1 + (1 - alpha) * ssim


# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=40, device="cuda"):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for masked_fft, unmasked_fft in dataloader:
                masked_fft = masked_fft.to(device)
                unmasked_fft = unmasked_fft.to(device)

                masked_fft = masked_fft.squeeze(2)
                unmasked_fft = unmasked_fft.squeeze(2)

                outputs = model(masked_fft)

                # Apply combined loss
                loss = criterion(outputs, unmasked_fft)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_time = epoch_duration * (num_epochs - (epoch + 1))
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f}s, "
              f"Estimated remaining time: {remaining_time / 60:.2f} minutes. "
              f"Average Loss: {running_loss / len(dataloader):.4f}")

    print("Training completed.")


# Main Function
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    masked_dir = r"D:\Desmoking Dataset\LH_frames\masked"
    unmasked_dir = r"D:\Desmoking Dataset\LH_frames\1970_01_01_010226_LH_S6\Smoke_clear_compare\clear"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(masked_dir, unmasked_dir, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)

    model = FFTRestorationNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Use combined loss
    train_model(model, dataloader, combined_loss, optimizer, num_epochs=40, device=device)

    model_path = r"D:\Desmoking Dataset\LH_frames\fft_restoration_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

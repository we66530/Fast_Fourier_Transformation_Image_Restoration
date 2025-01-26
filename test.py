import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the same model architecture as used in training
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

# Load the pre-trained model with weights_only=True
def load_model(model_path, device="cuda"):
    model = FFTRestorationNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the masked image and convert it to FFT format
def preprocess_image(image_path, transform=None):
    image = Image.open(image_path).convert("L")  # Convert to grayscale

    if transform:
        image = transform(image)

    # Perform FFT and stack real/imaginary parts
    image_fft = torch.fft.fft2(image)
    image_fft = torch.stack([image_fft.real, image_fft.imag], dim=0)  # Shape: [2, H, W]

    return image_fft.unsqueeze(0)  # Add batch dimension: [1, 2, H, W]

# Postprocess the model output and convert it back to an image
def postprocess_output(output):
    real_output = output[0, 0, :, :]  # Extract real part
    imag_output = output[0, 1, :, :]  # Extract imaginary part
    complex_output = torch.complex(real_output, imag_output)

    # Perform inverse FFT
    output_image = torch.fft.ifft2(complex_output)
    output_image = output_image.real  # Take only the real part

    return output_image

# Restore a new masked image
def restore_masked_image(model, image_path, device="cuda"):
    # Define the image preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to 256x256
        transforms.ToTensor()          # Convert to tensor
    ])

    # Preprocess the image
    image_fft = preprocess_image(image_path, transform)
    image_fft = image_fft.to(device)

    # Remove extra dimension if present
    if image_fft.dim() == 5:  # Check if there's an extra dimension
        image_fft = torch.squeeze(image_fft, dim=2)  # Squeeze the third dimension

    # Forward pass through the model
    with torch.no_grad():
        output = model(image_fft)

    # Postprocess the output
    restored_image = postprocess_output(output)

    # Convert the restored image to a NumPy array for PIL
    restored_image_np = restored_image.cpu().numpy()
    restored_image_np = (restored_image_np - restored_image_np.min()) / (
        restored_image_np.max() - restored_image_np.min()
    ) * 255.0  # Normalize to 0-255
    restored_image_np = restored_image_np.astype(np.uint8)  # Convert to uint8
    restored_pil_image = Image.fromarray(restored_image_np)  # Convert to PIL Image

    # Resize the image to (1080, 1920)
    resized_image = restored_pil_image.resize((1920, 1080), Image.BICUBIC)

    return resized_image

# Main function to run the restoration
if __name__ == "__main__":
    # Paths to the model and masked image
    model_path = r"D:\Desmoking Dataset\LH_frames\fft_restoration_model.pth"
    masked_image_path = r"D:\Desmoking Dataset\LH_frames\masked\frames_0000138.jpg"  # Replace with the new image path

    # Check for device (CUDA or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model = load_model(model_path, device=device)

    # Restore the masked image
    restored_image = restore_masked_image(model, masked_image_path, device=device)

    # Show the restored image using PIL
    restored_image.show()

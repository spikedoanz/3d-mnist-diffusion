import torch
import torchvision
from torchvision import transforms
import numpy as np
import nibabel as nib


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def create_noised_image(x, noise_step, total_steps=50):
    """
    Create a noised image at a specific noise step using cosine schedule.
    
    Args:
    x (torch.Tensor): Source image tensor of shape [1, 1, 256, 256, 256]
    noise_step (int): The specific noise step to generate (0 <= noise_step < total_steps)
    total_steps (int): Total number of steps in the noise schedule
    
    Returns:
    torch.Tensor: Noised image with shape [1, 1, 256, 256, 256]
    """
    device = x.device
    
    # Create beta schedule
    betas = cosine_beta_schedule(total_steps).to(device)
    
    # Calculate alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # Generate Gaussian noise
    noise = torch.randn_like(x)
    
    # Apply noise for the specific step
    noised_image = sqrt_alphas_cumprod[noise_step] * x + sqrt_one_minus_alphas_cumprod[noise_step] * noise
    
    return noised_image

def create_3d_mnist_digit(cube_size=28):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    idx = torch.randint(0, len(mnist), (1,)).item()
    digit, label = mnist[idx]
    
    digit_3d = digit.repeat(cube_size, 1, 1)
    digit_3d = digit_3d.unsqueeze(0).unsqueeze(0)
    
    print(f"Selected digit: {label}")
    return digit_3d

def save_as_nifti(tensor, filename):
    # Convert to numpy array and squeeze out batch and channel dimensions
    array = tensor.squeeze().cpu().numpy()
    
    # Create NIfTI image
    nifti_image = nib.Nifti1Image(array, affine=np.eye(4))
    
    # Save as NIfTI file
    nib.save(nifti_image, filename)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_3d = create_3d_mnist_digit().to(device)
    print(f"MNIST 3D tensor shape: {mnist_3d.shape}")
    
    # Set up noising parameters
    timesteps = 50
    noise_steps = [0, 8, 16, 32]  # Include 0 for the original image
    
    # Save original and noised images at different steps
    for step in noise_steps:
        if step == 0:
            image = mnist_3d
            filename = 'step0.nii.gz'
        else:
            image = create_noised_image(mnist_3d, step, total_steps=timesteps)
            filename = f'step{step}.nii.gz'
        
        save_as_nifti(image, filename)
        print(f"Saved {filename}")

    print("NIfTI files creation complete. You can now view them in your 3D image viewer tool.")

if __name__ == "__main__":
    main()

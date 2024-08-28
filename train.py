import os
import argparse

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

import torchvision
from torchvision import transforms

import nibabel as nib
from meshnet_3D import MeshNet3D, ExponentialMovingAverage


# Model configs
config_path     = 'meshnet_3D_config.json'
ema_decay       = 0.995
in_channels     = 1
out_channels    = 1
channels        = 128

volume_size     = (8, 28, 28)
cube_size       = volume_size[0]
timesteps       = 100

learning_rate   = 1e-5
batch_size      = 1
num_epochs      = 4000
num_samples     = 2**13

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


"""
DDP stuff
"""
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup(): torch.distributed.destroy_process_group()
"""
"""

def create_3d_mnist_dataset(num_samples=num_samples, cube_size=cube_size, label=-1):
    """
    Noisifies mnist dataset, pads some extra 0s on the z axis to give something to learn
    in that space and fills the rest with the stacked mnist image.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    labels = mnist if label == -1 else [item for item in mnist if item[1] == label]

    dataset = []
    for _ in range(num_samples):
        idx = torch.randint(0, len(labels), (1,)).item()
        digit, _ = labels[idx]
        digit_3d = torch.zeros(cube_size, 28, 28)           # Create the 3D digit with zero padding
        digit_repeated = digit.repeat(cube_size - 4, 1, 1)  # Repeat for inner slices
        digit_3d[2:cube_size-2] = digit_repeated            # Place repeated digit in the middle
        dataset.append(digit_3d.unsqueeze(0))
    return torch.stack(dataset)


def diffusion_loss_fn(model, x_0, t, betas):
    """
    Calculates L2 loss
    """
    alphas = 1. - betas                                                                     # Calculate alphas and related quantities
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1) # Get the corresponding values for the current timestep
    noise = torch.randn_like(x_0)                                                           # Generate noise
    x_noisy = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise         # Create noisy image
    predicted_noise = model(x_noisy, t)                                                     # Predict noise
    return nn.functional.mse_loss(noise, predicted_noise)
    

@torch.no_grad()
def sample(model, n_samples=1, device=torch.device("cuda")):
    """
    Runs inference for the learned backward diffusion function
    """
    model.eval()
    x = torch.randn(n_samples, 1, *volume_size).to(device)
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    for i in tqdm(reversed(range(timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        betas_t = betas[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1. / alphas[t]).view(-1, 1, 1, 1, 1)
        
        model_output = model(x, t)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if i > 0:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(betas_t) * noise
        else:
            x = model_mean
    
    model.train()
    return x


def save_as_nifti(tensor, filename):
    """
    Nifies are convenient for viewing 3d data https://brainder.org/2012/09/23/the-nifti-file-format/
    """
    array = tensor.squeeze().cpu().numpy()
    nifti_image = nib.Nifti1Image(array, affine=np.eye(4))
    nib.save(nifti_image, filename)


def load_checkpoint(model, optimizer, ema_model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Warning: No model state dict found in checkpoint. Attempting to load checkpoint directly.")
            model.load_state_dict(checkpoint)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: No optimizer state found in checkpoint.")
        if 'ema_model_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            print("Warning: No EMA model state found in checkpoint.")
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
    else:
        print("Warning: Checkpoint is not a dict. Attempting to load it directly into the model.")
        model.load_state_dict(checkpoint)
        start_epoch = 0
        best_loss = float('inf')
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return start_epoch, best_loss


def train_loop(model, dataloader, optimizer, device, rank=-1, checkpoint_path=None):
    is_distributed = rank != -1
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    ema_model = ExponentialMovingAverage(model.module if is_distributed else model, decay=ema_decay, device=device)
    betas = cosine_beta_schedule(timesteps).to(device)

    start_epoch = 0
    best_loss = float('inf')
    if checkpoint_path:
        start_epoch, best_loss = load_checkpoint(model.module if is_distributed else model, optimizer, ema_model, checkpoint_path)
        print(f"Resuming from epoch {start_epoch} with best loss {best_loss}")

    for epoch in range(start_epoch, num_epochs):
        if is_distributed:
            dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, disable=is_distributed and rank != 0)

        total_loss = 0
        num_batches = 0

        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            loss = diffusion_loss_fn(model, images, t, betas)

            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model)

            total_loss += loss.item()
            num_batches += 1

            if not is_distributed or rank == 0:
                pbar.set_postfix(MSE=loss.item())

        avg_loss = total_loss / num_batches

        if not is_distributed or rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'best_loss': best_loss,
                }
                torch.save(checkpoint, "best_checkpoint.pth")
                print(f"New best loss {best_loss}. Saved best_checkpoint.pth")

            if (epoch + 1) % 10 == 0:
                samples = sample(ema_model, n_samples=5, device=device)
                for j, sample_img in enumerate(samples):
                    save_as_nifti(sample_img, f"epoch_{epoch+1}_{j+1}.nii.gz")

    if is_distributed: 
        cleanup()


def run_training(rank, world_size, use_ddp, continue_training):
    if use_ddp:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = create_3d_mnist_dataset()

    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(TensorDataset(dataset, torch.zeros(dataset.shape[0])), 
                                batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(TensorDataset(dataset, torch.zeros(dataset.shape[0])), 
                                batch_size=batch_size, shuffle=True)

    model = MeshNet3D(in_channels, out_channels, channels, config_path).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = "best_checkpoint.pth" if continue_training else None
    train_loop(model, dataloader, optimizer, device, rank if use_ddp else -1, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MeshNet3D DDPM')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue training from best_checkpoint.pth')
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        print("MPS is available. Running in test mode with local MPS.")
        run_training(0, 1, use_ddp=False, continue_training=args.continue_training)
    else:
        print("MPS is not available. Running with Distributed Data Parallel.")
        world_size = torch.cuda.device_count()
        mp.spawn(run_training, args=(world_size, True, args.continue_training), nprocs=world_size, join=True)

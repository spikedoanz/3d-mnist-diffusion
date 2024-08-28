import torch
import torch.nn as nn
import math
import json

class MeshNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels, config_file):
        super(MeshNet3D, self).__init__()
        with open(config_file, "r") as f:
            config = json.load(f)
        
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = nn.ModuleList()
        
        # Initial convolution to go from in_channels to channels
        self.initial_conv = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        
        for i, layer_config in enumerate(config["layers"]):
            self.layers.append(MeshNetBlock3D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=layer_config["kernel_size"],
                stride=layer_config["stride"],
                padding=layer_config["padding"],
                dilation=layer_config["dilation"],
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                channels=channels
            ))
        
        # Final convolution to go from channels to out_channels
        self.final_conv = nn.Conv3d(channels, out_channels, kernel_size=1)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x, time):
        t = self.time_mlp(time)
        
        x = self.initial_conv(x)
        
        for layer in self.layers:
            x = layer(x, t)
        
        x = self.final_conv(x)
        
        return x

class MeshNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout_p, bnorm, gelu, channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.norm = nn.BatchNorm3d(out_channels) if bnorm else nn.Identity()
        self.act = nn.GELU() if gelu else nn.ReLU()
        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, t):
        h = self.conv(x)
        h = self.norm(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb.view(*time_emb.shape, 1, 1, 1)
        h = h + time_emb
        
        h = self.act(h)
        h = self.dropout(h)
        
        return h

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=True)

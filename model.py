import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

from diffwave.params import params as base_params  # 引入全局参数作为后备

##########################################################
# 噪声预测器（NoisePredictor）
##########################################################

class NoisePredictor(nn.Module):
    
    def __init__(self, params, max_t=50, embed_dim=64):
        super().__init__()

        self.embed_dim = embed_dim
        # 也可换用正弦位置编码, 这里直接 nn.Embedding
        self.t_embed = nn.Embedding(max_t+1, embed_dim)

        # 输入通道: 音频 1 + 时间步嵌入 embed_dim
        in_channels = 1 + embed_dim

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(64)

        self.conv_out = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
       
        B, C, T = x.shape
        # 1) 对 t 做嵌入 => (B, embed_dim)
        t_emb = self.t_embed(t)  # => (B, embed_dim)
        # 2) 扩展到时间维度 => (B, embed_dim, T)
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, T)

        # 3) 拼接: [B, 1 + embed_dim, T]
        x_cat = torch.cat([x, t_emb], dim=1)

        # 4) 卷积 + BN + ReLU
        h = self.conv1(x_cat)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)

        out = self.conv_out(h)
        return out


##########################################################
# DiffWave 主扩散模型
##########################################################

def silu(x):
   
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.proj1 = nn.Linear(128, 512)
        self.proj2 = nn.Linear(512, 512)

    def _build_embedding(self, max_steps):
        
        steps = torch.arange(max_steps).unsqueeze(1)  # [max_steps,1]
        dims  = torch.arange(64).unsqueeze(0)         # [1,64]
        table = steps * (10.0**(dims*4.0/63.0))       # [max_steps, 64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # => [max_steps, 128]
        return table

    def forward(self, t):
      
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            # 简单线性插值
            low_idx = torch.floor(t).long()
            high_idx= torch.ceil(t).long()
            low = self.embedding[low_idx]
            high= self.embedding[high_idx]
            x = low + (high-low)*(t-low_idx)
        
        x = self.proj1(x)
        x = silu(x)
        x = self.proj2(x)
        x = silu(x)
        return x  # shape [B, 512]


class SpectrogramUpsampler(nn.Module):
  
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1, kernel_size=(3,32), stride=(1,16), padding=(1,8))
        self.conv2 = nn.ConvTranspose2d(1, 1, kernel_size=(3,32), stride=(1,16), padding=(1,8))

    def forward(self, x):
      
        x = x.unsqueeze(1)  # => [B,1,n_mels,frames]
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = x.squeeze(1)  # => [B, n_mels, T_audio]
        return x


class ResidualBlock(nn.Module):

    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2*residual_channels,
            kernel_size=3, stride=1,
            dilation=dilation, padding=dilation
        )
        self.diff_proj = nn.Linear(512, residual_channels)  # 512是DiffusionEmbedding输出维度

        if not uncond:
            self.cond_proj = nn.Conv1d(n_mels, 2*residual_channels, kernel_size=1)
        else:
            self.cond_proj = None

        self.out_proj  = nn.Conv1d(residual_channels, 2*residual_channels, kernel_size=1)

    def forward(self, x, diff_emb, conditioner=None):
   
        B, C, T = x.shape

        # 将diff_emb投影并 unsqueeze 在时间维度
        diff_emb_out = self.diff_proj(diff_emb).unsqueeze(-1)  # => [B, residual_channels, 1]
        y = x + diff_emb_out

        # dilated conv
        y = self.dilated_conv(y)  # => [B, 2*residual_channels, T]

        if self.cond_proj is not None and conditioner is not None:
            cond_out = self.cond_proj(conditioner)
            y = y + cond_out

        # y 分两半 => gate, filter
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        # output projection => 2*residual_channels
        out = self.out_proj(y)
        residual, skip = torch.chunk(out, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):

    def __init__(self, params, noise_predictor=None, use_pnp=False):
        super().__init__()
        self.params = params

        self.input_projection = nn.Conv1d(1, params.residual_channels, kernel_size=1)
        self.diff_emb = DiffusionEmbedding(max_steps=len(params.noise_schedule))
        
        if self.params.unconditional:
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.res_blocks = nn.ModuleList()
        for i in range(params.residual_layers):
            dilation = 2 ** (i % params.dilation_cycle_length)
            block = ResidualBlock(
                n_mels=params.n_mels,
                residual_channels=params.residual_channels,
                dilation=dilation,
                uncond=params.unconditional
            )
            self.res_blocks.append(block)

        self.skip_proj = nn.Conv1d(params.residual_channels, params.residual_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(params.residual_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, audio, t, spectrogram=None):
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        x = self.input_projection(audio)
        x = F.relu(x)
        diff_emb = self.diff_emb(t)
        if self.spectrogram_upsampler:
            spectrogram = self.spectrogram_upsampler(spectrogram)
        skip_accum = None
        for layer in self.res_blocks:
            x, skip = layer(x, diff_emb, spectrogram)
            if skip_accum is None:
                skip_accum = skip
            else:
                skip_accum = skip_accum + skip
        x = skip_accum / sqrt(len(self.res_blocks))
        x = self.skip_proj(x)
        x = F.relu(x)
        x = self.out_proj(x)
        return x

    def fixed_reverse(self, x_tau, tau, spectrogram=None):
      
        noise_schedule = self.params.get("noise_schedule", base_params.noise_schedule)
        beta = np.array(noise_schedule)
        alpha = 1.0 - beta
        alpha_cum = np.cumprod(alpha)
        device = x_tau.device
        x = x_tau
        # 从 tau 逐步逆扩散到 0
        for t in range(tau, -1, -1):
            c1 = 1.0 / np.sqrt(alpha[t])
            c2 = beta[t] / np.sqrt(1.0 - alpha_cum[t])
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self(x, t_tensor, spectrogram=spectrogram)  # 传递 spectrogram
            x = c1 * (x - c2 * pred_noise)
            if t > 0:
                sigma = np.sqrt(beta[t])
                x = x + sigma * torch.randn_like(x)
            x = torch.clamp(x, -1.0, 1.0)
        return x

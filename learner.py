import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffwave.dataset import from_path, from_gtzan
from diffwave.model import DiffWave, NoisePredictor
from diffwave.params import AttrDict

def to_cpu(x):

    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_cpu(v) for v in x]
    else:
        return x

def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)

##############################################
# Multi-Scale STFT Loss 实现
##############################################
class MultiScaleSTFTLoss(nn.Module):

    def __init__(self, scales=[(1024, 256, 1024), (512, 128, 512), (256, 64, 256)], alpha=1.0, eps=1e-7):
        super().__init__()
        self.scales = scales
        self.alpha = alpha
        self.eps = eps

    def forward(self, x_pred, x_gt):
       
        B, C, T = x_pred.shape
        loss = 0.0
        for (win_len, hop_len, n_fft) in self.scales:
            # 将 [B, 1, T] 转换为 [B, T]
            spec_pred = self.stft(x_pred.view(B, T), n_fft, hop_len, win_len)
            spec_gt   = self.stft(x_gt.view(B, T), n_fft, hop_len, win_len)
            # L1 距离
            l_mag = torch.mean(torch.abs(spec_pred - spec_gt))
            loss += l_mag
        loss = loss / len(self.scales)
        return loss * self.alpha

    def stft(self, x, n_fft, hop_length, win_length):
      
        window = torch.hann_window(win_length, device=x.device)
        # stft 结果 shape: [B, freqs, frames, 2]
        X = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=True, return_complex=False, pad_mode='reflect'
        )
        # 计算幅度谱
        real = X[..., 0]
        imag = X[..., 1]
        mag = torch.sqrt(real**2 + imag**2 + self.eps)
        return mag

##############################################
# DiffWaveLearner 类
##############################################
class DiffWaveLearner:
     
    def __init__(self, model_dir, model, dataset, optimizer, params,
                 noise_predictor=None, use_pnp=False, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.params = params

        self.noise_predictor = noise_predictor  # 如果启用了 --use_noise_predictor，就会传入一个实例
        self.use_pnp = use_pnp  # 是否在训练中使用 PnP 加噪策略

        # 混合精度
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))

        self.step = 0
        self.is_master = True  # 用于分布式时判断是否是主进程

        # 计算 noise_level 供加噪时查询
        # noise_schedule 通常长度=50 => alpha_cum[t] = ∏_{i=1}^t (1-β_i)
        beta = np.array(self.params.noise_schedule)
        self.noise_level = np.cumprod(1 - beta).astype(np.float32)  # shape=[50]

        self.loss_fn = nn.L1Loss()

        # 初始化多尺度 STFT 损失模块
        self.ms_stft_loss = MultiScaleSTFTLoss()

        self.summary_writer = None
        self.grad_norm = None

    def state_dict(self):
        
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        return {
            'step': self.step,
            'model': {k: v.cpu() for k, v in model_state.items()},
            'optimizer': to_cpu(self.optimizer.state_dict()),
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
       
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = os.path.join(self.model_dir, save_basename)
        link_name = os.path.join(self.model_dir, f'{filename}.pt')
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        
        try:
            ckpt_path = os.path.join(self.model_dir, f'{filename}.pt')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' not in checkpoint:
                raise RuntimeError("Checkpoint file corrupted or missing 'model' key!")
            print(f"[INFO] Loading checkpoint from {ckpt_path}")
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_steps=None):
        
        device = next(self.model.parameters()).device
        while True:
            if self.is_master:
                with tqdm(
                    total=len(self.dataset),
                    desc=f'Epoch {self.step // len(self.dataset)}',
                    dynamic_ncols=True, position=0, leave=True,
                    mininterval=1, file=sys.stderr
                ) as pbar:
                    for features in self.dataset:
                        if max_steps is not None and self.step >= max_steps:
                            return
                        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                        loss = self.train_step(features)
                        tqdm.write(f"[step {self.step:>7}] loss = {loss.item():.6f}")
                        if torch.isnan(loss).any():
                            raise RuntimeError(f"[ERROR] NaN loss at step {self.step}")
                        if self.step % 50 == 0:
                            self._write_summary(self.step, features, loss)
                        if self.step % len(self.dataset) == 0:
                            self.save_to_checkpoint()
                        self.step += 1
                        pbar.update(1)
                        pbar.set_postfix(loss=float(loss))
            else:
                for features in self.dataset:
                    if max_steps is not None and self.step >= max_steps:
                        return
                    features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                    loss = self.train_step(features)
                    self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio']
        spectrogram = features['spectrogram']
        device = audio.device
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        N, _, T = audio.shape
        noise_level_torch = torch.from_numpy(self.noise_level).to(device=device)

        with self.autocast:
            total_steps = len(self.params.noise_schedule)
            t_np = np.random.randint(0, total_steps, size=N)
            t = torch.tensor(t_np, device=device, dtype=torch.long)
            alpha_cum_t = noise_level_torch.gather(0, t)
            alpha_cum_t_sqrt = alpha_cum_t.sqrt().view(-1, 1, 1)
            base_noise = torch.randn_like(audio)
            noisy_audio = alpha_cum_t_sqrt * audio + (1 - alpha_cum_t).sqrt().view(-1, 1, 1) * base_noise

            # 此处可根据需要开启或关闭 PnP 策略，目前注释掉
            # if (self.noise_predictor is not None) and self.use_pnp:
            #     predicted_noise = self.noise_predictor(noisy_audio, t)
            #     current_pnp_steps = max(
            #         self.params.pnp_steps_final,
            #         int(self.params.pnp_steps_initial * (self.params.pnp_decay_rate ** (self.step // 1000)))
            #     )
            #     prob_pnp = max(
            #         self.params.pnp_probability_end,
            #         self.params.pnp_probability_start * (self.params.pnp_decay_rate ** (self.step // 1000))
            #     )
            #     valid_mask = (t < current_pnp_steps).float().view(-1, 1, 1)
            #     rand_vals = torch.rand(N, device=device).view(-1, 1, 1)
            #     prob_mask = (rand_vals < prob_pnp).float()
            #     overall_mask = valid_mask * prob_mask
            #     noisy_audio = noisy_audio + overall_mask * predicted_noise

            pred_noise = self.model(noisy_audio, t, spectrogram)
            if pred_noise.dim() == 4:
                pred_noise = pred_noise.squeeze(2)
            # 原始噪声预测损失
            noise_loss = self.loss_fn(pred_noise, base_noise)
            # 利用预测噪声重构波形：
            # noisy_audio = sqrt(alpha_cum_t)*audio + sqrt(1 - alpha_cum_t)*base_noise
            # 若模型预测噪声与 base_noise 对应，则可重构为：
            pred_audio = (noisy_audio - (1 - alpha_cum_t).sqrt().view(-1,1,1) * pred_noise) / alpha_cum_t_sqrt
            # 时域 L1 损失
            waveform_l1 = F.l1_loss(pred_audio, audio)
            # 多尺度 STFT 损失
            stft_loss = self.ms_stft_loss(pred_audio, audio)
            # 合并波形相关损失（你可以调整这里的权重，下面给出默认权重）
            loss_lambda_stft = self.params.get("loss_lambda_stft", 1.0)
            combined_wave_loss = waveform_l1 + loss_lambda_stft * stft_loss
            # 最终总损失
            loss_lambda_noise = self.params.get("loss_lambda_noise", 1.0)
            loss_lambda_wave = self.params.get("loss_lambda_wave", 1.0)
            total_loss = loss_lambda_noise * noise_loss + loss_lambda_wave * combined_wave_loss

            loss = total_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.params.training_step += 1
        return loss

    def _write_summary(self, step, features, loss):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_scalar('train/loss', loss, step)
        if self.grad_norm is not None:
            writer.add_scalar('train/grad_norm', self.grad_norm, step)
        audio_0 = features['audio'][0]
        writer.add_audio('feature/audio', audio_0, step, sample_rate=self.params.sample_rate)
        if (not self.params.unconditional) and (features['spectrogram'] is not None):
            spec_0 = features['spectrogram'][0]
            spec_0 = spec_0.unsqueeze(0)
            writer.add_image('feature/spectrogram', spec_0, step, dataformats='CHW')
        writer.flush()
        self.summary_writer = writer

# ------------------ 新增噪声预测器训练函数 ------------------
def train_noise_predictor(args, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(args.model_dir, 'weights.pt')
    if not os.path.exists(ckpt_path):
        raise RuntimeError("请先训练主模型并保存 checkpoint 到 model_dir。")
    checkpoint = torch.load(ckpt_path, map_location=device)
    main_model = DiffWave(params).to(device)
    main_model.load_state_dict(checkpoint['model'], strict=False)
    main_model.eval()
    for param in main_model.parameters():
        param.requires_grad = False  # 冻结主扩散模型的所有参数

    fixed_diffusion_model = copy.deepcopy(main_model)
    for param in fixed_diffusion_model.parameters():
        param.requires_grad = False
    fixed_diffusion_model.eval()

    noise_predictor = NoisePredictor(params).to(device)
    optimizer_noise_pred = torch.optim.Adam(noise_predictor.parameters(), lr=2e-4)

    dataset = from_path(args.data_dirs, params)
    dataloader = dataset  # 使用 from_path 构建的 DataLoader，确保返回 'audio' 和 'gt'
    time_steps = params.noise_predictor_timesteps  # 例如 [12, 10, 7, 5]

    num_epochs = 100  # 可根据需要调整
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        noise_predictor.train()
        for features in tqdm(dataloader, desc=f"Noise Predictor Epoch {epoch}"):
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            audio = features['audio']
            gt = features['gt']
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            if gt.dim() == 2:
                gt = gt.unsqueeze(1)
            B = audio.shape[0]
            # tau = int(np.random.choice(time_steps))
            # 假设 time_steps = [12, 10, 7, 5]
            tau = time_steps[epoch % len(time_steps)]

            t_tensor = torch.full((B,), tau, device=device, dtype=torch.long)
            
            beta = np.array(params.noise_schedule)
            alpha = 1.0 - beta
            alpha_bar = np.cumprod(alpha)
            sqrt_alpha_bar_scalar = np.sqrt(alpha_bar[tau])
            sqrt_one_minus_alpha_bar_scalar = np.sqrt(1.0 - alpha_bar[tau])
            sqrt_alpha_bar = torch.tensor(np.full((B,), sqrt_alpha_bar_scalar), device=device, dtype=audio.dtype).view(B, 1, 1)
            sqrt_one_minus_alpha_bar = torch.tensor(np.full((B,), sqrt_one_minus_alpha_bar_scalar), device=device, dtype=audio.dtype).view(B, 1, 1)
            base_noise = torch.randn_like(audio)
            x_tau_standard = sqrt_alpha_bar * audio + sqrt_one_minus_alpha_bar * base_noise

            predicted_noise = noise_predictor(audio, t_tensor)
            # 这里直接将预测噪声加到标准噪声中构造 x_tau
            x_tau = x_tau_standard + predicted_noise

            spectrogram = features.get('spectrogram', None)
            if spectrogram is not None and spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(1)
            x_hat = fixed_diffusion_model.fixed_reverse(x_tau, tau, spectrogram=spectrogram)
            loss = nn.L1Loss()(x_hat, gt)
            optimizer_noise_pred.zero_grad()
            loss.backward()
            optimizer_noise_pred.step()
            # pbar.set_postfix(loss=float(loss))
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
        save_path = os.path.join(args.model_dir, f"noise_predictor_epoch{epoch}.pth")
        torch.save({'noise_predictor': noise_predictor.state_dict()}, save_path)





# ------------------ 原有 train() 和 train_distributed() ------------------
def _train_impl(replica_id, model, dataset, args, params, noise_predictor=None, use_pnp=False):
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    learner = DiffWaveLearner(
        model_dir=args.model_dir,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        params=params,
        noise_predictor=noise_predictor,
        use_pnp=use_pnp,
        fp16=args.fp16
    )
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)
    if replica_id == 0:
        print(model)
        if noise_predictor is not None:
            print(noise_predictor)

def train(args, params, noise_predictor=None, use_pnp=False):
    if len(args.data_dirs) > 0 and args.data_dirs[0] == 'gtzan':
        dataset = from_gtzan(params)
    else:
        dataset = from_path(args.data_dirs, params)
    model = DiffWave(params).cuda()
    _train_impl(
        replica_id=0,
        model=model,
        dataset=dataset,
        args=args,
        params=params,
        noise_predictor=noise_predictor,
        use_pnp=use_pnp
    )

def train_distributed(replica_id, replica_count, port, args, params,
                      noise_predictor=None, use_pnp=False):
    import torch.distributed as dist
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    if len(args.data_dirs) > 0 and args.data_dirs[0] == 'gtzan':
        dataset = from_gtzan(params, is_distributed=True)
    else:
        dataset = from_path(args.data_dirs, params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = DiffWave(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(
        replica_id=replica_id,
        model=model,
        dataset=dataset,
        args=args,
        params=params,
        noise_predictor=noise_predictor,
        use_pnp=use_pnp
    )

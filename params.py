 
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                self.__dict__[k] = v
        elif isinstance(attrs, (list, tuple, set)):
            for a in attrs:
                self.override(a)
        elif attrs is not None:
            raise NotImplementedError
        return self

params = AttrDict(
    # -----------------------------
    # Training params
    # -----------------------------
    batch_size = 2,
    learning_rate = 2e-4,
    max_grad_norm = None,

    # -----------------------------
    # Data params
    # -----------------------------
    sample_rate = 22050,
    n_mels = 80,
    n_fft = 1024,
    hop_samples = 256,
    crop_mel_frames = 62,  # 训练时随机裁剪的梅尔帧数

    # -----------------------------
    # Model params
    # -----------------------------
    residual_layers = 30,
    residual_channels = 64,
    dilation_cycle_length = 10,
    unconditional = False,  # 是否为无条件模型 (无 spectrogram)

    # -----------------------------
    # Diffusion noise schedule
    # -----------------------------
    noise_schedule = np.linspace(1e-4, 0.05,50).tolist(),

    # 推理时可选用的噪声 schedule (可选)
    inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # -----------------------------
    # For unconditional generation
    # -----------------------------
    audio_len = 22050 * 5,  # 5秒音频

    # -----------------------------
    # Noise predictor & PnP
    # -----------------------------
    use_noise_predictor = True,
    noise_predictor_hidden_dim = 128,
    noise_predictor_layers = 3,
    use_pnp = True,

    # 预定义的关键时间步 (用于噪声预测器训练阶段)
    noise_predictor_timesteps = [6,4,2],

    # PnP 相关参数 (动态阈值 / 概率衰减)
    pnp_steps_initial = 15,
    pnp_steps_final = 5,
    pnp_decay_rate = 0.99,
    pnp_strategy = "adaptive",
    pnp_probability_start = 0.9,
    pnp_probability_end = 0.2,

    # 记录全局训练步数, 可用于动态调整 PnP 参数
    training_step = 0
)

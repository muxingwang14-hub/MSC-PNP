# MSC-PnP: DiffWave-based Speech Generation with Noise Prediction

[![License](https://img.shields.io/badge/License-MIT-blue.svg )](LICENSE)

A DiffWave-based speech generation project that integrates **NoisePredictor** and **Partial Noise Prediction (PnP)** strategies for high-quality speech synthesis tasks.

---

## 📂 Repository Structure

```text
mscpnp/
├── __init__.py              # Package initialization
├── __main__.py              # Main program entry point
├── model.py                 # DiffWave model and NoisePredictor implementation
├── learner.py               # Training logic and loss functions
├── dataset.py               # Dataset loading and preprocessing
├── preprocess.py            # Audio preprocessing utilities
├── inference.py             # Inference and evaluation scripts
└── params.py                # Model parameter configuration
```

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/muxingwang14-hub/MSC-PNP.git
   cd MSC-PNP
   ```

2. **Install Python dependencies**
   ```bash
   pip install torch torchaudio transformers datasets jiwer sacremoses tqdm
   pip install librosa pystoi pesq matplotlib scipy
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

---

## 💡 Usage

### 1. Data Preprocessing

First, preprocess your audio data to generate mel-spectrograms:

```bash
python preprocess.py /path/to/audio/directory
```

### 2. Model Training

#### Main Model Training
```bash
python -m mscpnp /path/to/model/dir /path/to/audio/dirs --phase main --use_noise_predictor --use_pnp
```

#### Noise Predictor Training
```bash
python -m mscpnp /path/to/model/dir /path/to/audio/dirs --phase noise
```

#### Training Parameters
- `--phase`: Training phase (`main` or `noise`)
- `--use_noise_predictor`: Enable noise predictor
- `--use_pnp`: Enable Partial Noise Prediction strategy
- `--fp16`: Use half-precision training
- `--max_steps`: Maximum training steps

### 3. Model Inference

#### Single File Inference
```bash
python inference.py /path/to/model/dir --spectrogram_path /path/to/spec.npy --output output.wav --use_noise_predictor --use_pnp
```

#### Batch Testing
```bash
python inference.py /path/to/model/dir --pairs_list test_pairs.txt --test --use_noise_predictor --use_pnp
```

#### Inference Parameters
- `--spectrogram_path`: Mel-spectrogram path
- `--output`: Output audio file
- `--use_noise_predictor`: Use noise predictor
- `--use_pnp`: Use PnP strategy
- `--start_step`: PnP starting step
- `--test`: Batch testing mode

---

## 🔧 Key Features

### 1. DiffWave Model
- Diffusion model-based speech generation
- Supports conditional generation (based on mel-spectrograms)
- Configurable noise scheduling and network architecture

### 2. Noise Predictor (NoisePredictor)
- Lightweight CNN network for noise prediction
- Works collaboratively with the main model to improve generation quality
- Supports timestep embedding

### 3. Partial Noise Prediction (PnP) Strategy
- Adaptive noise prediction probability
- Gradually reduces from full diffusion process to partial prediction
- Balances generation quality and computational efficiency

### 4. Multi-Scale STFT Loss
- Combines L1 loss with multi-scale spectral loss
- Improves audio quality and spectral fidelity

---

## 📊 Evaluation Metrics

The project supports various audio quality evaluation metrics:
- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **WER/CER**: Word Error Rate/Character Error Rate (requires ASR model)

---

## ⚙️ Configuration Parameters

Main configuration parameters in `params.py`:

```python
params = AttrDict(
    # Training parameters
    batch_size = 2,
    learning_rate = 2e-4,
    
    # Audio parameters
    sample_rate = 22050,
    n_mels = 80,
    audio_len = 22050 * 5,  # 5 seconds of audio
    
    # Model parameters
    residual_layers = 30,
    residual_channels = 64,
    
    # Noise predictor parameters
    use_noise_predictor = True,
    noise_predictor_timesteps = [6, 4, 2],
    
    # PnP strategy parameters
    use_pnp = True,
    pnp_steps_initial = 15,
    pnp_steps_final = 5,
    pnp_decay_rate = 0.99,
)
```

---

## 🔬 Technical Details

### Diffusion Process
- Uses DDPM (Denoising Diffusion Probabilistic Models) framework
- 50-step noise scheduling with customizable inference steps
- Conditional generation based on mel-spectrogram upsampling

### Noise Predictor Architecture
- 1D CNN network with timestep embedding
- Input: Raw audio + timestep embedding
- Output: Predicted noise signal

### PnP Strategy
- Dynamically adjusts noise prediction probability
- Gradually decays from high probability (0.9) to low probability (0.2)
- Supports both adaptive and fixed decay strategies

---

## 📚 References

- **DiffWave**: [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761 )
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239 )
- **Wav2Vec2**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477 )

---

## 🤝 Contributing

Contributions and issues are welcome! Please ensure:

1. Code follows project style
2. Add appropriate tests
3. Update relevant documentation

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions or suggestions, please contact us via:

- Submit a GitHub Issue
- Send email to: [your-email@example.com]

---

## 🙏 Acknowledgments

Thanks to the following open-source projects:
- [DiffWave](https://github.com/lmnt-com/diffwave )
- [Hugging Face Transformers](https://github.com/huggingface/transformers )
- [Librosa](https://github.com/librosa/librosa )
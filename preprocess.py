 
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
import torchaudio


from diffwave.params import params







def transform(filename):
   
    try:
        # 读取音频
        audio, sr = T.load(filename)
        # 只取第 0 通道，确保单通道
        audio = audio[0]
        audio = torch.clamp(audio, -1.0, 1.0)


        if sr != params.sample_rate:
            resampler = TT.Resample(orig_freq=sr, new_freq=params.sample_rate)
            audio = resampler(audio)
            sr = params.sample_rate
            torchaudio.save(filename, audio.unsqueeze(0), sr)
            print(f"[Info] Overwrote {filename} with {sr} Hz resampled audio")
            #raise ValueError(f"Invalid sample rate {sr}, expected {params.sample_rate}.")

        # 定义梅尔谱转换参数
        mel_args = {
            'sample_rate': sr,
            'win_length': params.hop_samples * 4,
            'hop_length': params.hop_samples,
            'n_fft': params.n_fft,
            'f_min': 20.0,
            'f_max': sr / 2.0,
            'n_mels': params.n_mels,
            'power': 1.0,
            'normalized': True,
        }
        mel_spec_transform = TT.MelSpectrogram(**mel_args)

        with torch.no_grad():
            # 计算梅尔谱
            spec = mel_spec_transform(audio)
            # 对数转换
            spec = 20.0 * torch.log10(torch.clamp(spec, min=1e-5)) - 20.0
            # 归一化到 [0,1]
            spec = torch.clamp((spec + 100.0) / 100.0, 0.0, 1.0)
        
        # 直接保存为 .spec.npy
        np.save(f"{filename}.spec.npy", spec.cpu().numpy())
    
    except Exception as e:
        print(f"[Error] Processing {filename}: {e}")

def main(args):
   
    wav_files = glob(f"{args.dir}/**/*.wav", recursive=True)
    print(f"[INFO] Found {len(wav_files)} wav files in {args.dir}")
    if not wav_files:
        return

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform, wav_files), total=len(wav_files), desc="Preprocessing"))

if __name__ == '__main__':
    parser = ArgumentParser(description="Prepares a dataset by converting .wav files to Mel-Spectrograms")
    parser.add_argument("dir", help="Directory containing WAV files for training")
    args = parser.parse_args()
    main(args)


import matplotlib.pyplot as plt
import librosa.display
import os
print(">>> Running MODIFIED inference.py <<<")  
import time
import torch
import torchaudio
import numpy as np
from argparse import ArgumentParser
from pesq import pesq

import librosa                       # Mel-Spectrogram Distance / SegSNR
from scipy.signal import lfilter     # SegSNR
try:
    import vggish_input, vggish_params, vggish_slim
except ImportError:
    tf = None   

from pystoi import stoi
import torchaudio.functional as F

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave, NoisePredictor

import random
import numpy as np
import torch

from jiwer import wer, cer
import torchaudio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


LOCAL_HF_DIR = r"/data02/wz/diffwave/wav2vec2-base-960h" 


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def preprocess_for_metrics(clean, pred):

    clean = clean - np.mean(clean)
    pred  = pred  - np.mean(pred)

    rms_ref  = np.sqrt(np.mean(clean**2))
    rms_pred = np.sqrt(np.mean(pred**2))
    pred = pred * (rms_ref / (rms_pred+1e-8))
    return clean, pred


_loaded_models = {}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _load_model(model_dir, device):
    if model_dir in _loaded_models:
        return _loaded_models[model_dir]
    ckpt_path = os.path.join(model_dir, 'weights.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = torch.load(model_dir, map_location=device)
    model = DiffWave(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"[INFO] Model parameters: {count_parameters(model)/1e6:.2f}M")
    _loaded_models[model_dir] = model
    return model

@torch.no_grad()
def _do_reverse_diffusion(audio, model, start_step=None, spectrogram=None,
                          noise_predictor=None, use_pnp=False):
    beta = np.array(model.params.noise_schedule)
    alpha = 1.0 - beta
    alpha_cum = np.cumprod(alpha)
    device = audio.device

    t = start_step if start_step is not None else len(beta)-1
    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
    pred_noise_main = model(audio, t_tensor, spectrogram=spectrogram)
    if noise_predictor is not None and use_pnp:
        pred_noise_pnp = noise_predictor(audio, t_tensor)
        pred_noise = 0.5 * pred_noise_main + 0.5 * pred_noise_pnp
        
    else:
        pred_noise = pred_noise_main
    c1 = 1.0 / np.sqrt(alpha[t])
    c2 = beta[t] / np.sqrt(1.0 - alpha_cum[t])
    audio = c1 * (audio - c2 * pred_noise)
    if t > 0:
        sigma = np.sqrt((1.0 - alpha_cum[t-1]) / (1.0 - alpha_cum[t]) * beta[t])
        audio += sigma * torch.randn_like(audio)
    audio = torch.clamp(audio, -1.0, 1.0)


    for t_ in range(t-1, -1, -1):
        t_tensor = torch.tensor([t_], device=device, dtype=torch.long)
        pred_noise = model(audio, t_tensor, spectrogram=spectrogram)
        c1 = 1.0 / np.sqrt(alpha[t_])
        c2 = beta[t_] / np.sqrt(1.0 - alpha_cum[t_])
        audio = c1 * (audio - c2 * pred_noise)
        if t_ > 0:
            sigma = np.sqrt((1.0 - alpha_cum[t_-1]) / (1.0 - alpha_cum[t_]) * beta[t_])
            audio += sigma * torch.randn_like(audio)
        audio = torch.clamp(audio, -1.0, 1.0)
    return audio

@torch.no_grad()
def predict(model_dir: str, spectrogram=None, device=torch.device('cuda'),
            noise_predictor=None, use_pnp=False, params=None, start_step=None):
    if params is None:
        params = base_params
    model = _load_model(model_dir, device)
    model.params.override(params)
    model.eval()
    if noise_predictor is not None:
        noise_predictor.eval()

    if not model.params.unconditional:
        if spectrogram is not None and spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device) if spectrogram is not None else None
        batch_size = spectrogram.shape[0] if spectrogram is not None else 1
        time_len = model.params.hop_samples * \
                   (spectrogram.shape[-1] if spectrogram is not None else 62)
        audio = torch.randn(batch_size, 1, time_len, device=device)
    else:
        audio = torch.randn(1, 1, model.params.audio_len, device=device)


    out = _do_reverse_diffusion(audio, model,
                                start_step=start_step,
                                spectrogram=spectrogram,
                                noise_predictor=noise_predictor,
                                use_pnp=use_pnp)
    return out.squeeze(0)

def compute_snr(clean, pred):
    clean, pred = preprocess_for_metrics(clean, pred)
    noise = clean - pred
    return 10 * np.log10(
        np.sum(clean**2) / (np.sum(noise**2)+1e-8)
    )


def compute_rmse(clean, pred):
    return np.sqrt(np.mean((clean.flatten() - pred.flatten())**2))
import librosa
def compute_lsd(clean, pred, sr, win_len_ms=25, hop_ms=10):
    clean, pred = preprocess_for_metrics(clean, pred)
    n_fft = int(win_len_ms * sr / 1000)
    hop   = int(hop_ms   * sr / 1000)
    S_c = np.abs(librosa.stft(clean,
                              n_fft=n_fft,
                              hop_length=hop,
                              win_length=n_fft,
                              center=False))**2
    S_p = np.abs(librosa.stft(pred,
                              n_fft=n_fft,
                              hop_length=hop,
                              win_length=n_fft,
                              center=False))**2
    log_c = 10*np.log10(np.clip(S_c,1e-8,None))
    log_p = 10*np.log10(np.clip(S_p,1e-8,None))
    lsd_per_frame = np.sqrt(np.mean((log_c-log_p)**2, axis=0))
    return float(np.mean(lsd_per_frame))

def compute_pesq_resample(sr, ref, deg, target_sr=16000):
    if sr not in (8000, 16000):
        ref = F.resample(torch.tensor(ref), sr, target_sr).numpy()
        deg = F.resample(torch.tensor(deg), sr, target_sr).numpy()
        sr = target_sr
    mode = 'wb' if sr == 16000 else 'nb'
    return pesq(sr, ref, deg, mode)
def compute_spectral_convergence(clean, pred, n_fft=1024):
    mag_clean = np.abs(np.fft.rfft(clean.flatten(), n=n_fft))
    mag_pred  = np.abs(np.fft.rfft(pred.flatten(),  n=n_fft))
    return np.linalg.norm(mag_clean - mag_pred) / (np.linalg.norm(mag_clean) + 1e-8)
import librosa
def compute_mcd(clean, pred, sr, n_mfcc=13):
    clean, pred = preprocess_for_metrics(clean, pred)
    mfcc_c = librosa.feature.mfcc(y=clean, sr=sr, n_mfcc=n_mfcc)
    mfcc_p = librosa.feature.mfcc(y=pred,  sr=sr, n_mfcc=n_mfcc)
    dist = np.sqrt(np.sum((mfcc_c-mfcc_p)**2, axis=0))
    K = 10.0/np.log(10)*np.sqrt(2)
    return float(np.mean(K * dist))

def compute_si_sdr(clean, pred):
    c = clean.flatten() - np.mean(clean)
    p = pred.flatten() - np.mean(pred)
    scaling = np.dot(p, c)/(np.sum(c**2)+1e-8)
    target = scaling * c
    noise  = p - target
    return 10 * np.log10(np.sum(target**2)/ (np.sum(noise**2)+1e-8))
def compute_seg_snr(clean, pred, frame_len=0.03, sr=16000):

    clean, pred = clean.flatten(), pred.flatten()
    L = int(frame_len * sr)
    eps = 1e-8
    segsnr = []
    for i in range(0, len(clean)-L, L):
        c = clean[i:i+L]
        p = pred [i:i+L]
        noise = c - p
        segsnr.append(10*np.log10(np.sum(c**2)/(np.sum(noise**2)+eps)))
    return np.mean(segsnr) if segsnr else 0.0
def mel_spectrogram(x, sr=16000, n_fft=1024, hop=256, n_mels=80):
    return librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        power=1.0, center=False)

def compute_msd(clean, pred, sr=16000):
    S_c = mel_spectrogram(clean, sr)
    S_p = mel_spectrogram(pred , sr)
    S_c = librosa.amplitude_to_db(S_c, ref=1.0)
    S_p = librosa.amplitude_to_db(S_p, ref=1.0)
    min_T = min(S_c.shape[1], S_p.shape[1])
    diff = S_c[:,:min_T] - S_p[:,:min_T]
    return np.mean(np.abs(diff))


def compute_lsmse(clean, pred, n_fft=1024):


    clean_spec = librosa.stft(clean, n_fft=n_fft)
    pred_spec  = librosa.stft(pred, n_fft=n_fft)

    clean_log_mag = np.log1p(np.abs(clean_spec))
    pred_log_mag  = np.log1p(np.abs(pred_spec))
    

    mse = np.mean((clean_log_mag - pred_log_mag) ** 2)
    return mse

def compute_psnr(clean, pred):

    mse = np.mean((clean - pred) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 1.0 
    psnr = 10 * np.log10(max_pixel**2 / mse)
    return psnr



def get_gpu_memory_usage():
    return torch.cuda.max_memory_allocated()/(1024*1024) if torch.cuda.is_available() else 0.0

def test_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    processor = Wav2Vec2Processor.from_pretrained(
        LOCAL_HF_DIR,
        local_files_only=True
    )
    asr_model = Wav2Vec2ForCTC.from_pretrained(
        LOCAL_HF_DIR,
        local_files_only=True
    ).to(device)
    asr_model.eval()

    if args.pairs_list:

        with open(args.pairs_list) as f:
 
            triples = [l.strip().split('|') for l in f if l.strip()]
 
            pairs = [(s, w, t) for s,w,t in triples]
    else:
        if not (args.spectrogram_path and args.gt_audio):
            raise ValueError("need --spectrogram_path and --gt_audio")
        pairs = [(args.spectrogram_path,args.gt_audio)]


    if args.use_noise_predictor and args.use_pnp:
        configs = {
            "with_np_pnp": {
                "use_noise_predictor": True,
                "use_pnp": True,
                "start_step": args.start_step
            }
        }
    else:
        configs = {
            "without_np_pnp": {
                "use_noise_predictor": False,
                "use_pnp": False,
                "start_step": len(base_params.noise_schedule) - 1
            }
        }
    results = {k:{m:[] for m in ["time","snr","pesq","stoi","lsd","spectral_conv","mcd","rmse","si_sdr","seg_snr","msd", "wer","cer",     "gpu_mem"]} for k in configs}
    successful_count = 0
    for k, cfg in configs.items():
        print(f"\nTesting configuration: {k}")
        npd = NoisePredictor(base_params).to(device) if cfg["use_noise_predictor"] else None

        if npd is not None:

            ckpt_path = args.noise_predictor_ckpt \
                        if args.noise_predictor_ckpt else \
                        os.path.join(args.model_dir, 'noise_predictor_best.pth')
            print(f"[INFO] Loading NoisePredictor weights from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            npd.load_state_dict(ckpt['noise_predictor'])
            npd.eval()




        for spec_path, gt_path, gt_txt_path in pairs:
            print(f"[INFO] Processing spectrogram: {spec_path}, Ground truth: {gt_path}") 
            spec = torch.from_numpy(np.load(spec_path)).float()
            gt, sr = torchaudio.load(gt_path)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            pred = predict(args.model_dir,spectrogram=spec,device=device,
                           noise_predictor=npd,use_pnp=cfg["use_pnp"],
                           params=base_params,start_step=cfg["start_step"])
            elapsed = time.perf_counter()-t0
            results[k]["time"].append(elapsed)
            results[k]["gpu_mem"].append(get_gpu_memory_usage())
            pred_np = pred.cpu().numpy().flatten(); gt_np = gt.numpy().flatten()
            L = min(len(pred_np),len(gt_np))
            pred_np,gt_np = pred_np[:L],gt_np[:L]
            results[k]["snr"].append(compute_snr(gt_np,pred_np))
            results[k]["pesq"].append(compute_pesq_resample(sr,gt_np,pred_np))
            results[k]["stoi"].append(stoi(gt_np,pred_np,sr,extended=False))
            results[k]["lsd"].append(compute_lsd(gt_np,pred_np,sr))
            results[k]["spectral_conv"].append(compute_spectral_convergence(gt_np,pred_np))
            results[k]["mcd"].append(compute_mcd(gt_np,pred_np,sr))
            results[k]["rmse"].append(compute_rmse(gt_np,pred_np))
            results[k]["si_sdr"].append(compute_si_sdr(gt_np,pred_np))
            results[k]["seg_snr"].append(compute_seg_snr(gt_np, pred_np, sr=sr))
            results[k]["msd"].append(compute_msd(gt_np, pred_np, sr=sr))
                        # ==== 新增：ASR 转录 & WER/CER ====
            if gt_txt_path is not None:
              
                pred_tensor = torch.from_numpy(pred_np).unsqueeze(0)  
             
                pred_resampled = F.resample(pred_tensor, orig_freq=sr, new_freq=16000)
        
                audio_for_asr = pred_resampled.squeeze(0).numpy()
                inputs = processor(audio_for_asr, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = asr_model(inputs.input_values.to(device)).logits
                pred_ids = torch.argmax(logits, dim=-1)
                hypothesis = processor.batch_decode(pred_ids)[0].lower()

    
                with open(gt_txt_path, 'r', encoding='utf-8') as f_txt:
                    reference = f_txt.read().strip().lower()

      
                wer_score = wer(reference, hypothesis)
                cer_score = cer(reference, hypothesis)
                results[k]["wer"].append(wer_score)
                results[k]["cer"].append(cer_score)
                print(f"[METRICS] WER: {wer_score:.3f}, CER: {cer_score:.3f}")
                print(f"       Recognized: {hypothesis}")
                
            else:
     
                results[k]["wer"].append(None)
                results[k]["cer"].append(None)




            successful_count += 1
    for k,res in results.items():
        print(f"\nConfiguration: {k}")
        print(f"  平均推理时间: {np.mean(res['time']):.4f} s")
        print(f"  平均SNR: {np.mean(res['snr']):.2f} dB")
        print(f"  平均PESQ: {np.mean(res['pesq']):.2f}")
        print(f"  平均STOI: {np.mean(res['stoi']):.2f}")
        print(f"  平均LSD: {np.mean(res['lsd']):.4f}")
        print(f"  平均谱收敛: {np.mean(res['spectral_conv']):.4f}")
        print(f"  平均MCD: {np.mean(res['mcd']):.4f}")
        print(f"  平均RMSE: {np.mean(res['rmse']):.6f}")
        print(f"  平均SI-SDR: {np.mean(res['si_sdr']):.2f} dB")
        print(f"  平均SegSNR: {np.mean(res['seg_snr']):.2f} dB")
        print(f"  平均MSD  : {np.mean(res['msd']):.4f} dB")
        print("  FAD (需离线批量计算，可用 Google VGGish 工具。)")
        print(f"  平均GPU内存峰值: {np.mean(res['gpu_mem']):.2f} MB")
        print("[NOTE] MOS 需要主观听评获得。")

        wer_scores = [w for w in res['wer'] if w is not None]
        cer_scores = [c for c in res['cer'] if c is not None]
        if wer_scores:
            print(f"  平均WER: {np.mean(wer_scores)*100:.2f}%")
        if cer_scores:
            print(f"  平均CER: {np.mean(cer_scores)*100:.2f}%")
              
        print(f"       Recognized: {hypothesis}")
        print(f"       WER={results[k]['wer'][-1]*100:.2f}%  CER={results[k]['cer'][-1]*100:.2f}%")
    print(f"[INFO] Successfully processed {successful_count} data pairs.")



def main(args):
    if args.test:
        test_inference(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_predictor = NoisePredictor(base_params).to(device) if args.use_noise_predictor else None
 
    processor = Wav2Vec2Processor.from_pretrained(
    LOCAL_HF_DIR,
    local_files_only=True
)
    asr_model = Wav2Vec2ForCTC.from_pretrained(
        LOCAL_HF_DIR,
        local_files_only=True
    ).to(device)
    asr_model.eval()

    if noise_predictor is not None:
   
        ckpt = torch.load(os.path.join(args.model_dir,'noise_predictor_best.pth'),map_location=device)
        noise_predictor.load_state_dict(ckpt['noise_predictor'])
        noise_predictor.eval()
        print(f"[INFO] NoisePredictor parameters: {count_parameters(noise_predictor)/1e6:.2f}M")
        print(f"[INFO] Loaded NP weights from {os.path.join(args.model_dir,'noise_predictor_best.pth')}")
    if noise_predictor is not None:
  
        ckpt_path = args.noise_predictor_ckpt \
                    if args.noise_predictor_ckpt else \
                    os.path.join(args.model_dir, 'noise_predictor_best.pth')
        print(f"[INFO] Loading NoisePredictor weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        noise_predictor.load_state_dict(ckpt['noise_predictor'])
        noise_predictor.eval()
        print(f"[INFO] NoisePredictor parameters: {count_parameters(noise_predictor)/1e6:.2f}M")
        print(f"[INFO] Loaded NP weights from {ckpt_path}")




    spec_data = torch.from_numpy(np.load(args.spectrogram_path)).float() if args.spectrogram_path else None
    start_step = args.start_step if (args.use_noise_predictor and args.use_pnp) else len(base_params.noise_schedule)-1

    t0 = time.perf_counter()
    audio_out = predict(
        model_dir=args.model_dir,
        spectrogram=spec_data,
        device=device,
        noise_predictor=noise_predictor,
        use_pnp=args.use_pnp,
        params=base_params,
        start_step=start_step
    )
    infer_time = time.perf_counter() - t0
    print(f"[INFO] Inference time: {infer_time:.3f} s")
    if audio_out.dim() == 1:
        audio_out = audio_out.unsqueeze(0)
    print(f"[INFO] audio_out.shape = {audio_out.shape}")
    print(f"[DEBUG] Model output sample rate: {base_params.sample_rate} Hz")
    torchaudio.save(args.output, audio_out.cpu(), sample_rate=base_params.sample_rate)
    print(f"[INFO] Done. Saved to {args.output}")



    y = audio_out.cpu().numpy().flatten()

    S_out = librosa.feature.melspectrogram(
        y=y,
        sr=base_params.sample_rate,
        n_fft=1024,
        hop_length=base_params.hop_samples,
        n_mels=base_params.n_mels,
        power=1.0,
        center=False
    )

    log_S_out = librosa.power_to_db(S_out, ref=np.max)


    plt.figure(figsize=(6, 3))
    librosa.display.specshow(
        log_S_out,
        sr=base_params.sample_rate,
        hop_length=base_params.hop_samples,
        x_axis='time',
        y_axis='mel',
        fmin=0,
        fmax=base_params.sample_rate / 2,
        vmin=-80,    
        vmax=0
    )
    plt.title('Output Mel-Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()


    plt.savefig('output_msc.png', dpi=300)
    plt.close()




    if args.gt_audio:


        gt, sr = torchaudio.load(args.gt_audio)
        pred_np = audio_out.cpu().numpy().flatten()
        gt_np   = gt.numpy().flatten()
        L = min(len(pred_np), len(gt_np))
        pred_np, gt_np = pred_np[:L], gt_np[:L]
        print(f"[METRICS] SNR: {compute_snr(gt_np, pred_np):.2f} dB")
        try:
            p = compute_pesq_resample(sr, gt_np, pred_np)
        except:
            p = -1
        print(f"[METRICS] PESQ: {p:.2f}")
        print(f"[METRICS] STOI: {stoi(gt_np, pred_np, sr, extended=False):.2f}")
        print(f"[METRICS] LSD: {compute_lsd(gt_np, pred_np,sr):.4f}")
        print(f"[METRICS] SpecConv: {compute_spectral_convergence(gt_np, pred_np):.4f}")
        print(f"[METRICS] MCD: {compute_mcd(gt_np, pred_np,sr):.4f}")
        print(f"[METRICS] RMSE: {compute_rmse(gt_np, pred_np):.6f}")
        print(f"[METRICS] SI-SDR: {compute_si_sdr(gt_np, pred_np):.2f} dB")
        corr = np.corrcoef(gt_np, pred_np)[0,1]
        print(corr)  
        alpha = np.dot(gt_np, pred_np) / (np.dot(pred_np, pred_np) + 1e-8)
        pred_scaled = pred_np * alpha
        print('SNR after scale', compute_snr(gt_np, pred_scaled))
        if torch.cuda.is_available():
            print(f"[METRICS] GPU memory peak usage: {get_gpu_memory_usage():.2f} MB")
        print("[NOTE] MOS 需要通过主观听评获得。")
        print(f"[METRICS] SegSNR: {compute_seg_snr(gt_np, pred_np):.2f} dB")
        print(f"[METRICS] MSD   : {compute_msd(gt_np, pred_np):.4f} dB")
        print("  FAD (需离线批量计算，可用 Google VGGish 工具。)")
           
        if args.gt_audio and args.gt_txt:
           
            pred_tensor = torch.from_numpy(pred_np).unsqueeze(0)
            pred_resampled = F.resample(pred_tensor, orig_freq=sr, new_freq=16000)
            audio_for_asr = pred_resampled.squeeze(0).numpy()

         
            inputs = processor(audio_for_asr, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = asr_model(inputs.input_values.to(device)).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hypothesis = processor.batch_decode(pred_ids)[0].lower()

        
            with open(args.gt_txt, 'r', encoding='utf-8') as f:
                reference = f.read().strip().lower()

            wer_score = wer(reference, hypothesis) * 100
            cer_score = cer(reference, hypothesis) * 100
            print(f"[METRICS] WER: {wer_score:.2f}%")
            print(f"[METRICS] CER: {cer_score:.2f}%")

if __name__ == '__main__':
    parser = ArgumentParser(description='Run DiffWave inference (with optional accelerated sampling and additional metrics)')
    parser.add_argument('model_dir', help='Directory containing weights.pt')
    parser.add_argument('--spectrogram_path', '-s', help='Path to .spec.npy for conditional generation')
    parser.add_argument('--pairs_list', help='Txt file; each line: <spec.npy>|<gt.wav>，用于批量测试')
    parser.add_argument('--output', '-o', default='output.wav', help='Output file name')
    parser.add_argument('--use_noise_predictor', action='store_true', default=False,
                        help='Whether to load and use noise predictor in inference')
    parser.add_argument('--use_pnp', action='store_true', default=False,
                        help='Whether to use Partial Noise Prediction strategy in inference')
    parser.add_argument('--start_step', type=int, default=12,
                        help='If using noise predictor and PnP, start reverse diffusion from this step')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run in testing mode to compare configurations')
    parser.add_argument('--gt_audio', help='Path to ground truth audio for testing (required in test mode)')
    parser.add_argument('--gt_txt', help='Ground truth transcript file (.txt)')

    parser.add_argument('--num_tests', type=int, default=10,
                        help='Number of test runs for each configuration in test mode')
    parser.add_argument('--noise_predictor_ckpt', type=str, default=None,
                        help='路径可选，显式指定噪声预测器权重 .pth')
    args = parser.parse_args()
    main(args)

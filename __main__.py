
import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)
from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from diffwave.learner import train, train_distributed, train_noise_predictor
from diffwave.params import params
from diffwave.model import DiffWave, NoisePredictor

def _get_free_port():
    
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def main(args):
    
 
    replica_count = device_count()
    print(f"[INFO] GPU number is {replica_count}")

    if args.phase == 'main':
 
        noise_predictor = None
        if args.use_noise_predictor:
            noise_predictor = NoisePredictor(params).cuda()

        use_pnp = args.use_pnp

        if replica_count > 1:
            if params.batch_size % replica_count != 0:
                raise ValueError(
                    f"Batch size {params.batch_size} not divisible by #GPUs {replica_count}."
                )
            params.batch_size = params.batch_size // replica_count

            port = _get_free_port()
            spawn(
                train_distributed,
                args=(replica_count, port, args, params, noise_predictor, use_pnp),
                nprocs=replica_count,
                join=True
            )
        else:
            train(args, params, noise_predictor=noise_predictor, use_pnp=use_pnp)
    elif args.phase == 'noise':

        train_noise_predictor(args, params)
    else:
        raise ValueError("未知的训练阶段，请指定 --phase main 或 --phase noise")

if __name__ == '__main__':
    parser = ArgumentParser(description='Train (or resume training) a DiffWave model')

    parser.add_argument('model_dir',
                        help='Directory to store checkpoints and logs')
    parser.add_argument('data_dirs', nargs='+',
                        help='Directories from which to read .wav files')

    parser.add_argument('--max_steps', default=None, type=int,
                        help='Max training steps (optional)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use 16-bit float training')


    parser.add_argument('--use_noise_predictor', action='store_true', default=False,
                        help='Enable the NoisePredictor module in training')

    parser.add_argument('--use_pnp', action='store_true', default=False,
                        help='Enable Partial Noise Prediction strategy')


    parser.add_argument('--phase', default='main', choices=['main', 'noise'],
                        help='Training phase: main for main model training, noise for noise predictor training')

    args = parser.parse_args()

    print("\n========== DiffWave Training Configuration ==========")
    print(f"Model Directory    : {args.model_dir}")
    print(f"Data Directories   : {args.data_dirs}")
    print(f"Max Steps          : {args.max_steps if args.max_steps else 'Unlimited'}")
    print(f"FP16 Training      : {'Enabled' if args.fp16 else 'Disabled'}")
    print(f"Noise Predictor    : {'Enabled' if args.use_noise_predictor else 'Disabled'}")
    print(f"PnP Strategy       : {'Enabled' if args.use_pnp else 'Disabled'}")
    print(f"Training Phase     : {args.phase}")

    main(args)

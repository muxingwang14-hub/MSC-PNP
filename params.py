 
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

    batch_size = 2,
    learning_rate = 2e-4,
    max_grad_norm = None,

   
    sample_rate = 22050,
    n_mels = 80,
    n_fft = 1024,
    hop_samples = 256,
    crop_mel_frames = 62, 

    
    residual_layers = 30,
    residual_channels = 64,
    dilation_cycle_length = 10,
    unconditional = False, 

   
    noise_schedule = np.linspace(1e-4, 0.05,50).tolist(),

  
    inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    
    audio_len = 22050 * 5,  

 
    use_noise_predictor = True,
    noise_predictor_hidden_dim = 128,
    noise_predictor_layers = 3,
    use_pnp = True,


    noise_predictor_timesteps = [6,4,2],


    pnp_steps_initial = 15,
    pnp_steps_final = 5,
    pnp_decay_rate = 0.99,
    pnp_strategy = "adaptive",
    pnp_probability_start = 0.9,
    pnp_probability_end = 0.2,


    training_step = 0
)

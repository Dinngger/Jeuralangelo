import numpy as np
from collections.abc import Mapping


class Config(Mapping):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __setitem__(self, k, v):
        self.kwargs[k] = v
        setattr(self, k, v)
    def __getitem__(self, k):
        return self.kwargs[k]
    def pop(self, k, default=None):
        if k in self.kwargs:
            delattr(self, k)
            return self.kwargs.pop(k)
        else:
            return default
    def __len__(self):
        return len(self.kwargs)
    def __iter__(self):
        return iter(self.kwargs)
    def items(self):
        return self.kwargs.items()
    def values(self):
        return self.kwargs.values()

fast_train = True
logdir = "/tmp/jeuralangelo/"
logging_iter = 9999999999
max_epoch = 9999999999
max_iter = 50000 if fast_train else 200000

wandb_scalar_iter = 100
wandb_image_iter = 10000
validation_iter = 5000
speed_benchmark = False

checkpoint = Config(
    save_iter = 10000,
    save_latest_iter = 9999999999,
    save_epoch = 9999999999,
    save_period = 9999999999,
)

data = Config(
    root = '',
    sphere_center = np.array([0.0, 0.0, 0.0]),
    sphere_radius = 1.0,
    preload = True,
    num_workers = 4,
    train = Config(
        image_size = [1080, 1920],  # H, W
        batch_size = 1,
        subset = None
    ),
    val = Config(
        image_size = [270, 480],
        batch_size = 1,
        subset = 1,
        max_viz_samples = 16
    )
)

trainer = Config(
    ema_config = Config(
        enabled = False,
        load_ema_checkpoint = False
    ),
    loss_weight = Config(
        render = 1.0,
        # s3im = 1.0,
        # mask = 0.5,
        eikonal = 0.1,
        curvature = 5e-4
    ),
    init = Config(
        type = 'none',
        gain = None
    ),
    amp_config = Config(
        enabled = False
    ),
    grad_accum_iter = 1,
    depth_vis_scale = 0.5
)

optim = Config(
    type = "AdamW",
    params = Config(
        lr = 1e-3,
        weight_decay = 1e-2,
        # fused = True
    ),
    sched = Config(
        iteration_mode = True,
        type = "two_steps_with_warmup",
        warm_up_end = 5000,
        two_steps = [180000, 190000],
        gamma = 10.0,
    ),
)

model = Config(
    object = Config(
        sdf = Config(
            mlp = Config(
                num_layers = 1,
                hidden_dim = 256,
                skip = [],
                activ = 'softplus',
                activ_params = Config(
                    beta = 100
                ),
                geometric_init = True,
                weight_norm = True,
                out_bias = 0.5,
                inside_out = False,
            ),
            encoding = Config(
                type = 'hashgrid',
                levels = 16,
                fp16 = True,
                hashgrid = Config(
                    min_logres =  5,
                    max_logres =  11,
                    dict_size = 21 if fast_train else 22,
                    dim = 2 if fast_train else 8,
                    range =  [-2,2],
                ),
                coarse2fine = Config(
                    enabled = True,
                    init_active_level = 4,
                    step = 2500 if fast_train else 5000,
                )
            ),
            gradient = Config(
                mode = 'numerical',
                taps = 4,
            ),
        ),
        rgb = Config(
            mlp = Config(
                num_layers = 4,
                hidden_dim = 256,
                skip = [],
                activ = 'relu',
                activ_params = {},
                weight_norm = True,
            ),
            mode = 'idr',
            encoding_view = Config(
                type = 'spherical',
                levels = 3,
            ),
        ),
        s_var = Config(
            init_val = 3.,
            anneal_end = 0.1,
        ),
    ),
    background = Config(
        enabled = True,
        white = False,
        mlp = Config(
            num_layers = 8,
            hidden_dim = 256,
            skip = [4],
            num_layers_rgb = 2,
            hidden_dim_rgb = 128,
            skip_rgb = [],
            activ = 'relu',
            activ_params = {},
            activ_density = 'softplus',
            activ_density_params = {},
        ),
        view_dep = True,
        encoding = Config(
            type = 'fourier',
            levels = 10,
        ),
        encoding_view = Config(
            type = 'spherical',
            levels = 3,
        ),
    ),
    render = Config(
        rand_rays = 512,
        in_mask_rays_rate = 0.5,
        num_samples = Config(
            coarse = 64,
            fine = 16,
            background = 32,
        ),
        num_sample_hierarchy = 4,
        stratified = True,
    ),
    appear_embed = Config(   
        enabled = False,
        dim = 8,
    )
)

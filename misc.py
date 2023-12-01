
import os
import random
import collections
import threading
from functools import partial
import numpy as np
import time
import jittor as jt
import config


flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def get_optimizer(model):
    r"""Return the optimizer object.

    Args:
        cfg_optim (obj): Config for the specific optimization module (gen/dis).
        model (obj): Jittor network object.

    Returns:
        (obj): Jittor optimizer
    """
    cfg_optim = config.optim
    if hasattr(model, 'get_param_groups'):
        # Allow the network to use different hyperparameters (e.g., learning rate) for different parameters.
        params = model.get_param_groups()
    else:
        params = model.parameters()

    try:
        # Try the Jittor optimizer class first.
        optimizer_class = getattr(jt.optim, cfg_optim.type)
    except AttributeError:
        raise NotImplementedError(f"Optimizer {cfg_optim.type} is not yet implemented.")
    optimizer_kwargs = cfg_optim.params

    optim = optimizer_class(params, **optimizer_kwargs)

    return optim


def get_scheduler(opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): Jittor optimizer object.

    Returns:
        (obj): Scheduler
    """
    cfg_opt = config.optim
    if cfg_opt.sched.type == 'two_steps_with_warmup':
        warm_up_end = cfg_opt.sched.warm_up_end
        two_steps = cfg_opt.sched.two_steps
        gamma = cfg_opt.sched.gamma

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                if x > two_steps[1]:
                    return 1.0 / gamma ** 2
                elif x > two_steps[0]:
                    return 1.0 / gamma
                else:
                    return 1.0

        scheduler = jt.optim.LambdaLR(opt, lambda x: sch(x))
    elif cfg_opt.sched.type == 'cos_with_warmup':
        alpha = cfg_opt.sched.alpha
        max_iter = cfg_opt.sched.max_iter
        warm_up_end = cfg_opt.sched.warm_up_end

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                progress = (x - warm_up_end) / (max_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
                return learning_factor

        scheduler = jt.optim.LambdaLR(opt, lambda x: sch(x))
    else:
        raise NotImplementedError
    return scheduler


def nan_to_num(x):
    x[jt.logical_not(jt.isfinite(x))] = 0.0


def mask_loss(weights: jt.Var,    # [B,R,No+Nb,1]
              mask: jt.Var):      # [B,R,1]
    Nc, Nf, Nb = config.model.render.num_samples.values()
    No = Nc + Nf * config.model.render.num_sample_hierarchy
    weights_sum = weights[:, :, :No, :].sum(dim=2)  # [B,R,1]
    mask_loss = jt.nn.binary_cross_entropy_with_logits(weights_sum, mask)
    nan_to_num(mask_loss)
    return mask_loss.mean()


def eikonal_loss(gradients, outside=None):
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # [B,R,N]
    nan_to_num(gradient_error)  # [B,R,N]
    if outside is not None:
        return (gradient_error * jt.logical_not(outside).float()).mean()
    else:
        return gradient_error.mean()


def curvature_loss(hessian, outside=None):
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N]
    nan_to_num(laplacian)  # [B,R,N]
    if outside is not None:
        return (laplacian * jt.logical_not(outside).float()).mean()
    else:
        return laplacian.mean()


class SoftPlus(jt.Function):
    def execute(self, x, beta=1.0, threshold=20.0):
        self.save_vars = x, beta, threshold
        x_shape = x.shape
        x = jt.reshape(x, -1)
        x = jt.code(x.shape, x.dtype, [x],
            cpu_src=f'''
            for (int idx=0; idx<out_shape0; idx++) {{
                in0_type aop = @in0(idx);
                @out(idx) = aop * {beta} > {threshold} ? aop
                        : (std::log1p(std::exp(aop * {beta}))) / {beta};
            }}
            ''',
            cuda_src=f'''
                __global__ static void softplus(@ARGS_DEF) {{
                    @PRECALC
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= @out_shape0)
                        return;
                    in0_type aop = @in0(idx);
                    @out(idx) = aop * {beta} > {threshold} ? aop
                        : (::log1p(std::exp(aop * {beta}))) / {beta};
                }}
                int dim = @out_shape0 / 1024 + 1;
                softplus<<<dim, 1024>>>(@ARGS);
            ''')
        return jt.reshape(x, x_shape)
    def grad(self, grad_y):
        x, beta, threshold = self.save_vars
        x_shape = x.shape
        x = jt.reshape(x, -1)
        grad_y = jt.reshape(grad_y, -1)
        grad_x = jt.code(grad_y.shape, grad_y.dtype, [grad_y, x],
            cpu_src=f'''
            for (int idx=0; idx<out_shape0; idx++) {{
                in0_type aop = @in0(idx);
                in1_type bop = @in1(idx);
                in0_type z = std::exp(bop * {beta});
                @out(idx) = bop * {beta} > {threshold} ? aop
                    : aop * z / (z + 1.0);
            }}
            ''',
            cuda_src=f'''
                __global__ static void softplus_backward(@ARGS_DEF) {{
                    @PRECALC
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= @out_shape0)
                        return;
                    in0_type aop = @in0(idx);
                    in1_type bop = @in1(idx);
                    in0_type z = std::exp(bop * {beta});
                    @out(idx) = bop * {beta} > {threshold} ? aop
                        : aop * z / (z + 1.0);
                }}
                int dim = @out_shape0 / 1024 + 1;
                softplus_backward<<<dim, 1024>>>(@ARGS);
            ''')
        return jt.reshape(grad_x, x_shape)


# def softplus(x, beta=1.0, threshold=20.0):
#     return jt.where(x * beta > threshold, x, jt.log1p(jt.exp(x * beta)) / beta)
#     return 1 / beta * jt.log(1 + (beta * x).minimum(threshold).exp()) + \
#         (x - threshold/beta).maximum(0.0)


def get_activation(activ, **kwargs):
    if activ == 'softplus':
        assert 'threshold' not in kwargs
        if 'beta' in kwargs:
            return lambda x: SoftPlus()(x, kwargs['beta'])
        else:
            return SoftPlus()
    func = dict(
        identity=lambda x: x,
        relu=jt.nn.relu,
        abs=jt.abs,
        sigmoid=jt.sigmoid,
        exp=jt.exp,
        # softplus=softplus,
        silu=jt.nn.silu,
    )[activ]
    return partial(func, **kwargs)


def to_full_image(image, image_size=None, from_vec=True):
    # if from_vec is True: [B,HW,...,K] --> [B,K,H,W,...]
    # if from_vec is False: [B,H,W,...,K] --> [B,K,H,W,...]
    if from_vec:
        assert image_size is not None
        target_shape = image.shape
        target_shape = target_shape[0:1] + image_size + target_shape[2:]
        image = jt.reshape(image, target_shape)
    axis = list(range(len(image.shape)))
    axis = [axis[0], axis[-1]] + axis[1:-1]
    image = image.transpose(tuple(axis))
    return image


def requires_grad(model, require=True):
    r""" Set a model to require gradient or not.

    Args:
        model (nn.Module): Neural network model.
        require (bool): Whether the network requires gradient or not.

    Returns:

    """
    for p in model.parameters():
        p.requires_grad = require


def set_random_seed(seed):
    r"""Set random seeds for everything

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each gpu will use a different random seed.
    """
    print(f"Using random seed {seed}")
    # random.seed(seed)
    # np.random.seed(seed)
    jt.set_global_seed(seed)         # sets seed on the current CPU & all GPUs


class Checkpointer(object):

    def __init__(self, model, optim=None, sched=None):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.logdir = config.logdir
        self.save_period = config.checkpoint.save_period
        self.iteration_mode = config.optim.sched.iteration_mode
        self.resume = False
        self.resume_epoch = self.resume_iteration = None

    def save(self, current_epoch, current_iteration, latest=False):
        r"""Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            current_epoch (int): Current epoch.
            current_iteration (int): Current iteration.
            latest (bool): If ``True``, save it using the name 'latest_checkpoint.pt'.
        """
        checkpoint_file = 'latest_checkpoint.pt' if latest else \
                          f'epoch_{current_epoch:05}_iteration_{current_iteration:09}_checkpoint.pkl'
        save_dict = self._collect_state_dicts()
        save_dict.update(
            epoch=current_epoch,
            iteration=current_iteration,
        )
        self._save_worker(save_dict, checkpoint_file, 0)
        # Run the checkpoint saver in a separate thread.
        # threading.Thread(
        #     target=self._save_worker, daemon=False, args=(save_dict, checkpoint_file, 0)).start()
        checkpoint_path = self._get_full_path(checkpoint_file)
        return checkpoint_path

    def _save_worker(self, save_dict, checkpoint_file, rank=0):
        checkpoint_path = self._get_full_path(checkpoint_file)
        # Save to local disk.
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        jt.save(save_dict, checkpoint_path)
        if rank == 0:
            self.write_latest_checkpoint_file(checkpoint_file)
        print('Saved checkpoint to {}'.format(checkpoint_path))

    def _collect_state_dicts(self):
        r"""Collect all the state dicts from network modules to be saved."""
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict()
        )

    def load(self, checkpoint_path=None, resume=False, load_opt=True, **kwargs):
        r"""Load network weights, optimizer parameters, scheduler parameters from a checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint (local file or S3 key).
            resume (bool): if False, only the model weights are loaded. If True, the metadata (epoch/iteration) and
                           optimizer/scheduler (optional) are also loaded.
            load_opt (bool): Whether to load the optimizer state dict (resume should be True).
            load_sch (bool): Whether to load the scheduler state dict (resume should be True).
        """
        # Priority: (1) checkpoint_path (2) latest_path (3) train from scratch.
        self.resume = resume
        # If checkpoint path were not specified, try to load the latest one from the same run.
        if resume and checkpoint_path is None:
            latest_checkpoint_file = self.read_latest_checkpoint_file()
            if latest_checkpoint_file is not None:
                checkpoint_path = self._get_full_path(latest_checkpoint_file)
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            self.checkpoint_path = checkpoint_path
            state_dict = jt.load(checkpoint_path)
            print(f"Loading checkpoint (local): {checkpoint_path}")
            # Load the state dicts.
            print('- Loading the model...')
            self.model.load_state_dict(state_dict['model'])
            if resume:
                self.resume_epoch = state_dict['epoch']
                self.resume_iteration = state_dict['iteration']
                self.sched.last_epoch = self.resume_iteration if self.iteration_mode else self.resume_epoch
                if load_opt:
                    print('- Loading the optimizer...')
                    self.optim.load_state_dict(state_dict['optim'])
                print(f"Done with loading the checkpoint (epoch {self.resume_epoch}, iter {self.resume_iteration}).")
            else:
                print('Done with loading the checkpoint.')
            self.eval_epoch = state_dict['epoch']
            self.eval_iteration = state_dict['iteration']
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            print('Training from scratch.')
        jt.gc()

    def _get_full_path(self, file):
        return os.path.join(self.logdir, file)

    def _get_latest_pointer_path(self):
        return self._get_full_path('latest_checkpoint.txt')

    def read_latest_checkpoint_file(self):
        checkpoint_file = None
        latest_path = self._get_latest_pointer_path()
        if os.path.exists(latest_path):
            checkpoint_file = open(latest_path).read().strip()
            if checkpoint_file.startswith("latest_checkpoint:"):  # TODO: for backward compatibility, to be removed
                checkpoint_file = checkpoint_file.split(' ')[-1]
        return checkpoint_file

    def write_latest_checkpoint_file(self, checkpoint_file):
        latest_path = self._get_latest_pointer_path()
        content = f"{checkpoint_file}\n"
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'File not found (local): {checkpoint_path}')

    def reached_checkpointing_period(self, timer):
        if timer.checkpoint_toc() > self.save_period:
            print('checkpointing period!')
            return True
        return False


class Timer:
    def __init__(self):
        self.time_iteration = 0.0
        self.time_epoch = 0

    def checkpoint_tic(self):
        # reset timer
        self.checkpoint_start_time = time.time()

    def checkpoint_toc(self):
        # return time by minutes
        return (time.time() - self.checkpoint_start_time) / 60


def collate_test_data_batches(data_batches):
    """Aggregate the list of test data from all devices and process the results.
    Args:
        data_batches (list): List of (hierarchical) dictionaries, where leaf entries are tensors.
    Returns:
        data_gather (dict): (hierarchical) dictionaries, where leaf entries are concatenated tensors.
    """
    data_gather = dict()
    for key in data_batches[0].keys():
        data_list = [data[key] for data in data_batches]
        if isinstance(data_batches[0][key], dict):
            data_gather[key] = collate_test_data_batches(data_list)
        elif isinstance(data_batches[0][key], jt.Var):
            data_gather[key] = jt.concat(data_list, dim=0)
            data_gather[key] = jt.concat([data_gather[key].contiguous()], dim=0)
        else:
            raise TypeError
    return data_gather


def get_unique_test_data(data_gather, idx):
    """Aggregate the list of test data from all devices and process the results.
    Args:
        data_gather (dict): (hierarchical) dictionaries, where leaf entries are tensors.
        idx (tensor): sample indices.
    Returns:
        data_all (dict): (hierarchical) dictionaries, where leaf entries are tensors ordered by idx.
    """
    data_all = dict()
    for key, value in data_gather.items():
        if isinstance(value, dict):
            data_all[key] = get_unique_test_data(value, idx)
        elif isinstance(value, jt.Var):
            data_all[key] = []
            for i in range(max(idx) + 1):
                # If multiple occurrences of the same idx, just choose the first one. If no occurrence, just ignore.
                matches = (idx == i).nonzero()
                if matches.numel() != 0:
                    data_all[key].append(value[matches[0, 0]])
            data_all[key] = jt.concat(data_all[key], dim=0)
        else:
            raise TypeError
    return data_all

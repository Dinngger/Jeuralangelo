
import os
import time
import jittor as jt
import wandb
from tqdm import tqdm
import inspect
import config
from model import Model
from visualization import wandb_image
from misc import Checkpointer, Timer, get_optimizer, get_scheduler, \
    mask_loss, eikonal_loss, curvature_loss, requires_grad, set_random_seed, \
    collate_test_data_batches, get_unique_test_data
from init_weight import weights_init, weights_rescale
# from s3im import S3IM


class Trainer:

    def __init__(self, train_data_loader, eval_data_loader=None, is_inference=True, seed=0):
        self.model = self.setup_model(seed)
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.metrics = dict()
        if not is_inference:
            self.optim = get_optimizer(self.model)
            self.optim_zero_grad_kwargs = {}
            if 'set_to_none' in inspect.signature(self.optim.zero_grad).parameters:
                self.optim_zero_grad_kwargs['set_to_none'] = True
            self.sched = self.setup_scheduler(self.optim)
            self.init_logging_attributes()
        else:
            self.optim = None
            self.sched = None
        # Data loaders & inference mode.
        self.is_inference = is_inference
        # Initialize loss functions.
        self.init_losses()
        self.weights = {key: value for key, value in config.trainer.loss_weight.items() if value}

        self.checkpointer = Checkpointer(self.model, self.optim, self.sched)
        self.timer = Timer()

        self.warm_up_end = config.optim.sched.warm_up_end
        self.cfg_gradient = config.model.object.sdf.gradient
        self.c2f_step = config.model.object.sdf.encoding.coarse2fine.step
        self.model.neural_sdf.warm_up_end = self.warm_up_end

    def setup_model(self, seed=0):
        r"""Return the networks. We will first set the random seed to a fixed value so that each GPU copy will be
        initialized to have the same network weights. We will then use different random seeds for different GPUs.
        After this we will wrap the network with a moving average model if applicable.

        The following objects are constructed as class members:
          - model (obj): Model object (historically: generator network object).

        Args:
            cfg (obj): Global configuration.
            seed (int): Random seed.
        """
        # We first set the random seed for all the process so that we initialize each copy of the network the same.
        set_random_seed(seed)
        # Construct networks
        model = Model()
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model parameter count: {:,}'.format(model_size))
        print(f'Initialize model weights using type: {config.trainer.init.type}, gain: {config.trainer.init.gain}')
        init_bias = getattr(config.trainer.init, 'bias', None)
        init_gain = config.trainer.init.gain or 1.
        model.apply(weights_init(config.trainer.init.type, init_gain, init_bias))
        model.apply(weights_rescale())
        # Different GPU copies of the same model will receive noises initialized with different random seeds
        # (if applicable) thanks to the set_random_seed command (GPU #K has random seed = args.seed + K).
        set_random_seed(seed)
        return model

    def init_losses(self):
        r"""Initialize loss functions. All loss names have weights. Some have criterion modules."""
        self.losses = dict()

        # Mapping from loss names to criterion modules.
        self.criteria = jt.nn.ModuleList()
        # Mapping from loss names to loss weights.
        self.weights = dict()
        self.criteria.add_module("render", jt.nn.L1Loss())
        # self.criteria.add_module("s3im", S3IM())
        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))

    def init_logging_attributes(self):
        r"""Initialize logging attributes."""
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0

    def setup_scheduler(self, optim):
        return get_scheduler(optim)

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
            self.metrics["psnr"] = -10 * jt.nn.mse_loss(data["rgb"], data["image_sampled"]).log() / jt.log(10)
            if "s3im" in self.weights:
                self.losses["s3im"] = self.criteria["s3im"](data["rgb"], data["image_sampled"], mode)
            if "mask" in self.weights:
                self.losses["mask"] = mask_loss(data["weights"], data["mask_sampled"])
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            if "s3im" in self.weights:
                self.losses["s3im"] = self.criteria["s3im"](data["rgb_map"], data["image"], mode)
            self.metrics["psnr"] = -10 * jt.nn.mse_loss(data["rgb_map"], data["image"]).log() / jt.log(10)

    def get_curvature_weight(self, current_iteration, init_weight):
        if "curvature" in self.weights:
            if current_iteration <= self.warm_up_end:
                self.weights["curvature"] = current_iteration / self.warm_up_end * init_weight
            else:
                model = self.model
                decay_factor = model.neural_sdf.growth_rate ** (model.neural_sdf.anneal_levels - 1)
                self.weights["curvature"] = init_weight / decay_factor

    def _start_of_iteration(self, data, current_iteration):
        model = self.model
        self.progress = model.progress = current_iteration / config.max_iter
        if config.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                self.get_curvature_weight(current_iteration, config.trainer.loss_weight.curvature)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()

        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Things to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch

        # Accumulate time
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % config.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / config.logging_iter
            self.timer.time_iteration = avg_time
            print('Iteration: {}, average iter time: {:6f}.'.format(current_iteration, avg_time))
            self.elapsed_iteration_time = 0

            if config.speed_benchmark:
                # only needed when analyzing computation bottleneck.
                self.timer._print_speed_benchmark(avg_time)

        # Log to wandb.
        if current_iteration % config.wandb_scalar_iter == 0:
            # Compute the elapsed time (as in the original base trainer).
            self.timer.time_iteration = self.elapsed_iteration_time / config.wandb_scalar_iter
            self.elapsed_iteration_time = 0
            # Log scalars.
            self.log_wandb_scalars(data, mode="train")
            # Exit if the training loss has gone to NaN/inf.
            if self.losses["total"].isnan():
                self.finalize()
                raise ValueError("Training loss has gone to NaN!!!")
            if self.losses["total"].isinf():
                self.finalize()
                raise ValueError("Training loss has gone to infinity!!!")
        if current_iteration % config.wandb_image_iter == 0:
            self.log_wandb_images(data, mode="train")
        # Run validation on val set.
        if current_iteration % config.validation_iter == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to W&B.
            self.log_wandb_scalars(data_all, mode="val")
            self.log_wandb_images(data_all, mode="val", max_samples=config.data.val.max_viz_samples)

        # Save everything to the checkpoint by time period.
        if self.checkpointer.reached_checkpointing_period(self.timer):
            self.checkpointer.save(current_epoch, current_iteration)
            self.timer.checkpoint_tic()  # reset timer

        # Save everything to the checkpoint.
        if current_iteration % config.checkpoint.save_iter == 0 or \
                current_iteration == config.max_iter:
            self.checkpointer.save(current_epoch, current_iteration)

        # Save everything to the checkpoint using the name 'latest_checkpoint.pt'.
        if current_iteration % config.checkpoint.save_latest_iter == 0:
            if current_iteration >= config.checkpoint.save_latest_iter:
                self.checkpointer.save(current_epoch, current_iteration, True)

        # Update the learning rate policy for the generator if operating in the iteration mode.
        if config.optim.sched.iteration_mode:
            self.sched.step()

    def log_wandb_scalars(self, data, mode=None):
        scalars = dict()
        # Log scalars (basic info & losses).
        if mode == "train":
            scalars.update({"optim/lr": self.sched.get_last_lr()[0]})
            scalars.update({"time/iteration": self.timer.time_iteration})
            scalars.update({"time/epoch": self.timer.time_epoch})
        scalars.update({f"{mode}/loss/{key}": value.item() for key, value in self.losses.items()})
        scalars.update(iteration=self.current_iteration, epoch=self.current_epoch)
        wandb.log(scalars, step=self.current_iteration)

        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach().item(),
            f"{mode}/s-var": self.model.s_var.item(),
        }
        if "curvature" in self.weights:
            scalars[f"{mode}/curvature_weight"] = self.weights["curvature"]
        if "eikonal" in self.weights:
            scalars[f"{mode}/eikonal_weight"] = self.weights["eikonal"]
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            scalars[f"{mode}/epsilon"] = self.model.neural_sdf.normal_eps
        if config.model.object.sdf.encoding.coarse2fine.enabled:
            scalars[f"{mode}/active_levels"] = self.model.neural_sdf.active_levels
        wandb.log(scalars, step=self.current_iteration)

    def log_wandb_images(self, data, mode=None, max_samples=None):
        images = {"iteration": self.current_iteration, "epoch": self.current_epoch}
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()  # * data["mask"]
            images.update({
                f"{mode}/vis/rgb_target": wandb_image(data["image"]),
                f"{mode}/vis/rgb_render": wandb_image(data["rgb_map"]),
                f"{mode}/vis/rgb_error": wandb_image(images_error),
                f"{mode}/vis/normal": wandb_image(data["normal_map"], from_range=(-1, 1)),
                f"{mode}/vis/inv_depth": wandb_image(1 / (data["depth_map"] + 1e-8) * config.trainer.depth_vis_scale),
                f"{mode}/vis/opacity": wandb_image(data["opacity_map"]),
            })
        wandb.log(images, step=self.current_iteration)

    def train(self, show_pbar=False):
        self.progress = self.model.progress = self.current_iteration / config.max_iter

        self.current_epoch = self.checkpointer.resume_epoch or self.current_epoch
        self.current_iteration = self.checkpointer.resume_iteration or self.current_iteration
        if (self.current_iteration % config.validation_iter == 0):
            # Do an initial validation.
            data_all = self.test(self.eval_data_loader, mode="val", show_pbar=show_pbar)
            # Log the results to W&B.
            self.log_wandb_scalars(data_all, mode="val")
            self.log_wandb_images(data_all, mode="val", max_samples=config.data.val.max_viz_samples)

        # Train.
        start_epoch = self.checkpointer.resume_epoch or self.current_epoch  # The epoch to start with.
        current_iteration = self.checkpointer.resume_iteration or self.current_iteration  # The starting iteration.

        self.timer.checkpoint_tic()  # start timer
        for current_epoch in range(start_epoch, config.max_epoch):
            self.current_epoch = current_epoch
            self.start_epoch_time = time.time()
            if show_pbar:
                data_loader_wrapper = tqdm(self.train_data_loader, desc=f"Training epoch {current_epoch + 1}", leave=False)
            else:
                data_loader_wrapper = self.train_data_loader
            for it, data in enumerate(data_loader_wrapper):
                data = self._start_of_iteration(data, current_iteration)
                self.current_iteration = current_iteration
                self.model.train()
                self.start_iteration_time = time.time()

                requires_grad(self.model, True)
                output = self.model(data)
                data.update(output)
                # Compute loss.
                self._compute_loss(data, mode="train")
                total_loss = jt.array(0.)
                # Iterates over all possible losses.
                for loss_name in self.weights:
                    if loss_name in self.losses:
                        # Multiply it with the corresponding weight and add it to the total loss.
                        total_loss += self.losses[loss_name] * self.weights[loss_name]
                self.losses['total'] = total_loss  # logging purpose
                # Scale down the loss w.r.t. gradient accumulation iterations.
                total_loss = total_loss / float(config.trainer.grad_accum_iter)
                self.optim.backward(total_loss)

                # Perform an optimizer step. This enables gradient accumulation when grad_accum_iter is not 1.
                last_iter_in_epoch=(it == len(self.train_data_loader) - 1)
                if (self.current_iteration + 1) % config.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
                    self.optim.step()
                    # Zero out the gradients.
                    self.optim.zero_grad(**self.optim_zero_grad_kwargs)

                # Update model average.
                if config.trainer.ema_config.enabled:
                    self.model.module.update_average()

                for loss_name in self.losses:
                    self.losses[loss_name] = self.losses[loss_name].detach()

                current_iteration += 1
                if show_pbar:
                    data_loader_wrapper.set_postfix(iter=current_iteration)
                if it == len(self.train_data_loader) - 1:
                    self.end_of_iteration(data, current_epoch + 1, current_iteration)
                else:
                    self.end_of_iteration(data, current_epoch, current_iteration)
                if current_iteration >= config.max_iter:
                    print('Done with training!!!')
                    return

            self.current_iteration = current_iteration
            self.current_epoch = current_epoch
            if not config.optim.sched.iteration_mode:
                self.sched.step()
            elapsed_epoch_time = time.time() - self.start_epoch_time
            # Logging.
            print('Epoch: {}, total time: {:6f}.'.format(current_epoch, elapsed_epoch_time))
            self.timer.time_epoch = elapsed_epoch_time

            # Save everything to the checkpoint.
            if current_epoch % config.checkpoint.save_epoch == 0:
                self.checkpointer.save(current_epoch, current_iteration)
        print('Done with training!!!')

    @jt.no_grad()
    def test(self, data_loader, output_dir=None, inference_args=None, mode="test", show_pbar=False):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if config.trainer.ema_config.enabled:
            model = self.model.averaged_model
        else:
            model = self.model
        model.eval()
        if show_pbar:
            data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        for it, data in enumerate(data_loader):
            data = self._start_of_iteration(data, self.current_iteration)
            self.model.train()
            self.start_iteration_time = time.time()
            output = model.inference(data)
            data.update(output)
            data_batches.append(data)
        # Aggregate the data from all devices and process the results.
        data_gather = collate_test_data_batches(data_batches)
        # Only the master process should process the results; slaves will just return.
        data_all = get_unique_test_data(data_gather, data_gather["idx"])
        tqdm.write(f"Evaluating with {len(data_all['idx'])} samples.")
        # Validate/test.
        if mode == "val":
            self._compute_loss(data_all, mode=mode)
            total_loss = jt.array(0.)
            # Iterates over all possible losses.
            for loss_name in self.weights:
                if loss_name in self.losses:
                    # Multiply it with the corresponding weight and add it to the total loss.
                    total_loss += self.losses[loss_name] * self.weights[loss_name]
            self.losses['total'] = total_loss  # logging purpose
        if mode == "test":
            # Dump the test results for postprocessing.
            self.dump_test_results(data_all, output_dir)
        return data_all

    def init_wandb(self, logdir, wandb_id=None, project="", run_name=None, mode="online", resume="allow", use_group=False):
        r"""Initialize Weights & Biases (wandb) logger.

        Args:
            cfg (obj): Global configuration.
            wandb_id (str): A unique ID for this run, used for resuming.
            project (str): The name of the project where you're sending the new run.
                If the project is not specified, the run is put in an "Uncategorized" project.
            run_name (str): name for each wandb run (useful for logging changes)
            mode (str): online/offline/disabled
        """
        print('Initialize wandb')
        if not wandb_id:
            wandb_path = os.path.join(logdir, "wandb_id.txt")
            wandb_id = wandb.util.generate_id()
            with open(wandb_path, "w") as f:
                f.write(wandb_id)
        if use_group:
            group, name = logdir.split("/")[-2:]
        else:
            group, name = None, os.path.basename(logdir)

        if run_name is not None:
            name = run_name

        wandb.init(id=wandb_id,
                    project=project,
                    group=group,
                    name=name,
                    dir=logdir,
                    resume=resume,
                    settings=wandb.Settings(start_method="fork"),
                    mode=mode)
        # if self.model is not None:
        #     wandb.watch(self.model)

    def finalize(self):
        # Finish the W&B logger.
        wandb.finish()

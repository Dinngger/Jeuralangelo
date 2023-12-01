
import argparse
import os
import sys
import jittor as jt
import config
import numpy as np
from functools import partial
from dataset import Dataset
from trainer import Trainer
from mesh import extract_texture, extract_mesh
jt.flags.use_cuda = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset_path', type=str, help='Dataset directory with cameras_sphere.npz and image/ and mask/ folders in it.')
    parser.add_argument('--logdir', type=str, help='Log directory.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--wandb', type=str, default="online", help="using Weights & Biases as the logger, options: online/offline/disabled")
    parser.add_argument('--wandb_name', default='default', type=str)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def train(args):
    dataset = Dataset(args.dataset_path)
    val_dataset = Dataset(args.dataset_path, is_inference=True)
    val_data_loader = jt.dataset.DataLoader(val_dataset,
                                            batch_size=config.data.val.batch_size)
    data_loader = jt.dataset.DataLoader(dataset, batch_size=config.data.train.batch_size)
    trainer = Trainer(data_loader, val_data_loader, is_inference=False)
    os.makedirs(config.logdir, exist_ok=True)
    trainer.init_wandb(logdir=config.logdir,
                       project=args.wandb_name,
                       mode=args.wandb,
                       use_group=True)
    trainer.train()
    trainer.finalize()


def mesh(textured=True, resolution=2048, block_res=128):
    # Initialize data loaders and models.
    trainer = Trainer(None, is_inference=True)
    # Load checkpoint.
    checkpoint = trainer.checkpointer.read_latest_checkpoint_file()
    checkpoint = trainer.checkpointer._get_full_path(checkpoint)
    trainer.checkpointer.load(checkpoint, load_opt=False)
    trainer.model.eval()

    # Set the coarse-to-fine levels.
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if config.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model.neural_sdf.set_active_levels(trainer.current_iteration - 1)
        print(f"set active levels to {trainer.model.neural_sdf.active_levels}")
        if config.model.object.sdf.gradient.mode == "numerical":
            trainer.model.neural_sdf.set_normal_epsilon()

    bounds = (np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]) - np.expand_dims(config.data.sphere_center, axis=1)) / config.data.sphere_radius

    sdf_func = lambda x: -trainer.model.neural_sdf.sdf(x)  # noqa: E731
    texture_func = partial(extract_texture, neural_sdf=trainer.model.neural_sdf,
                           neural_rgb=trainer.model.neural_rgb,
                           appear_embed=trainer.model.appear_embed) if textured else None
    mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, resolution=resolution * config.data.sphere_radius,
                        block_res=block_res, texture_func=texture_func, filter_lcc=False)

    print(f"vertices: {len(mesh.vertices)}")
    print(f"faces: {len(mesh.faces)}")
    if textured:
        print(f"colors: {len(mesh.visual.vertex_colors)}")
    # center and scale
    mesh.vertices = mesh.vertices * config.data.sphere_radius + config.data.sphere_center
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.export(os.path.join(config.logdir, "latest.ply"))


if __name__ == "__main__":
    args, cfg_cmd = parse_args()
    dataset_path = args.dataset_path
    from PIL import Image
    img_size = Image.open(dataset_path + "/image/000.png").size     # Read any image file. Assume all images are the same size.
    config.logdir = args.logdir
    config.data.train.image_size = [img_size[1], img_size[0]]
    config.data.val.image_size = [img_size[1] // 4, img_size[0] // 4]
    assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version must not be lower than sm_61!"
    train(args)
    mesh(resolution=1024 if config.fast_train else 2048)

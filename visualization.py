
import wandb
import jittor as jt
from jittor.transform import to_pil_image

from matplotlib import pyplot as plt


def wandb_image(images, from_range=(0, 1)):
    images = preprocess_image(images, from_range=from_range)
    image_grid = jt.make_grid(images, nrow=1, pad_value=1)
    if image_grid.ndim == 4 and image_grid.shape[0] == 1:
        image_grid = image_grid[0]
    image_grid = jt.transpose(image_grid, (1, 2, 0))
    image_grid = to_pil_image(image_grid)
    wandb_image = wandb.Image(image_grid)
    return wandb_image


def preprocess_image(images, from_range=(0, 1), cmap="gray"):
    min, max = from_range
    images = (images - min) / (max - min)
    images = images.detach().float().clamp_(min_v=0, max_v=1)
    if images.shape[1] == 1:
        images = get_heatmap(images[:, 0], cmap=cmap)
    return images


def get_heatmap(gray, cmap):  # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = jt.array(color[..., :3]).permute(0, 3, 1, 2).float()  # [N,3,H,W]
    return color

import os
from glob import glob
import numpy as np
import cv2 as cv
import jittor as jt
import jittor.transform as jt_trans
from PIL import Image, ImageFile
import camera, config

ImageFile.LOAD_TRUNCATED_IMAGES = True
import tqdm
import threading
import queue

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose[:3, :]

class Dataset(jt.dataset.Dataset):
    def __init__(self, dataset_dir, load_image=True, is_inference=False):
        super().__init__()
        self.data_dir = dataset_dir
        self.num_rays = config.model.render.rand_rays
        self.in_mask_num_rays = int(config.model.render.in_mask_rays_rate * self.num_rays)
        self.split = "val" if is_inference else "train"

        camera_dict = np.load(os.path.join(self.data_dir, 'cameras_sphere.npz'))
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.n_images = len(self.images_lis)
        if config.data[self.split].subset:
            subset = config.data[self.split].subset
            subset_idx = np.linspace(0, self.n_images, subset+1)[:-1].astype(int)
            self.images_lis = [self.images_lis[i] for i in subset_idx]
            self.masks_lis = [self.masks_lis[i] for i in subset_idx]
            self.n_images = len(self.images_lis)
        else:
            subset_idx = list(range(self.n_images))

        self.H, self.W = config.data.val.image_size if is_inference else config.data.train.image_size
        if load_image:
            self.images = self.preload_threading(self.get_image, 8, "images", self.images_lis)
        self.masks = self.preload_threading(self.get_image, 8, "masks", self.masks_lis)
        all_index = jt.arange(0, self.W * self.H)
        self.imasks = [all_index[jt_trans._to_jittor_array(mask)[0, :, :].flatten() > 0.5] for mask in self.masks]

        # world_mat is a projection matrix from world to image
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in subset_idx]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in subset_idx]
        self.cameras = []

        # auto scale.
        if not is_inference:
            radius = []
            for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, c2w = load_K_Rt_from_P(None, P)
                radius.append(np.linalg.norm(c2w[:3, -1]))
            print("min sphere radius: ", np.min(radius))
            print("average sphere radius: ", np.mean(radius))
            config.data.sphere_radius = min(np.mean(radius) * 0.7, config.data.sphere_radius)
            print("auto scale sphere radius: ", config.data.sphere_radius)

        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            c2w[:3, -1] -= config.data.sphere_center
            c2w[:3, -1] /= config.data.sphere_radius
            w2c = camera.Pose().invert(c2w)
            intrinsics, w2c = self.preprocess_camera(jt.array(intrinsics), w2c)
            self.cameras.append((intrinsics, w2c))

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_bbox_min = object_bbox_min[:, None]
        object_bbox_max = object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
    
    def __len__(self):
        return self.n_images

    def _preload_worker(self, data_list, load_func, name_list, q, lock, idx_tqdm):
        # Keep preloading data in parallel.
        while True:
            idx = q.get()
            data_list[idx] = load_func(name_list[idx])
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self, load_func, num_workers, data_str, name_list):
        # Use threading to preload data in parallel.
        n = len(name_list)
        data_list = [None] * n
        q = queue.Queue(maxsize=n)
        idx_tqdm = tqdm.tqdm(range(n), desc=f"preloading {data_str}", leave=False)
        for i in range(n):
            q.put(i)
        lock = threading.Lock()
        for ti in range(num_workers):
            t = threading.Thread(target=self._preload_worker,
                                 args=(data_list, load_func, name_list, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert all(map(lambda x: x is not None, data_list))
        return data_list

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image = self.preprocess_image(self.images[idx])
        mask = jt_trans._to_jittor_array(self.masks[idx])[0:1, :, :]
        imask = self.imasks[idx]
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx]
        # Pre-sample ray indices.
        if self.split == "train":
            out_mask_num_rays = self.num_rays - self.in_mask_num_rays
            ray_idx = imask[jt.randperm(imask.shape[0])[:self.in_mask_num_rays]]  # [Ri]
            ray_idx = jt.concat([ray_idx, jt.randperm(self.W * self.H)[:out_mask_num_rays]])  # [R]
            # ray_idx = jt.randperm(self.W * self.H)[:self.num_rays]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            mask_sampled = mask.flatten(1, 2)[:, ray_idx].t()   # [R,1]
            # image_sampled = image_sampled * mask_sampled
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                mask_sampled=mask_sampled,
                intr=intr,
                pose=pose,
            )
        else:  # keep image during inference
            sample.update(
                image=image,
                mask=mask,
                intr=intr,
                pose=pose,
            )
        return sample

    def get_image(self, image_fname):
        image = Image.open(image_fname)
        image.load()
        if self.split != "train":
            image = image.resize((self.W, self.H))
        return image

    def preprocess_image(self, image):
        # Resize the image.
        image = jt_trans._to_jittor_array(image)
        rgb = image[:3]
        return rgb

    def preprocess_camera(self, intr, pose):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_H, raw_W = config.data.train.image_size
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr, pose

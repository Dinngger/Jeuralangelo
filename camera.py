import numpy as np
import jittor as jt


class Pose():
    """
    A class of operations on camera poses (tensors with shape [...,3,4]).
    Each [3,4] camera pose takes the form of [R|t].
    """

    def __call__(self, R=None, t=None):
        # Construct a camera pose from the given R and/or t.
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, jt.Var):
                t = jt.array(t)
            R = jt.init.eye(3).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, jt.Var):
                R = jt.array(R)
            t = jt.zeros(R.shape[:-1])
        else:
            if not isinstance(R, jt.Var):
                R = jt.array(R)
            if not isinstance(t, jt.Var):
                t = jt.array(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = jt.concat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # Invert a camera pose.
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # Compose a sequence of poses together.
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new


def to_hom(X):
    # Get homogeneous coordinates of the input.
    X_hom = jt.concat([X, jt.ones_like(X[..., :1])], dim=-1)
    return X_hom


# Basic operations of transforming 3D points between world/camera/image coordinates.
def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ jt.linalg.inv(cam_intr).transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


def get_center_and_ray(pose, intr, image_size):
    """
    Args:
        pose (tensor [3,4]/[B,3,4]): Camera pose.
        intr (tensor [3,3]/[B,3,3]): Camera intrinsics.
        image_size (list of int): Image size.
    Returns:
        center_3D (tensor [HW,3]/[B,HW,3]): Center of the camera.
        ray (tensor [HW,3]/[B,HW,3]): Ray of the camera with depth=1 (note: not unit ray).
    """
    H, W = image_size
    # Given the intrinsic/extrinsic matrices, get the camera center and ray directions.
    with jt.no_grad():
        # Compute image coordinate grid.
        y_range = jt.arange(H, dtype=jt.float32).add_(0.5)
        x_range = jt.arange(W, dtype=jt.float32).add_(0.5)
        Y, X = jt.meshgrid(y_range, x_range)  # [H,W]
        xy_grid = jt.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    # Compute center and ray.
    if len(pose.shape) == 3:
        batch_size = len(pose)
        xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
    grid_3D = img2cam(to_hom(xy_grid), intr)  # [HW,3]/[B,HW,3]
    center_3D = jt.zeros_like(grid_3D)  # [HW,3]/[B,HW,3]
    # Transform from camera to world coordinates.
    grid_3D = cam2world(grid_3D, pose)  # [HW,3]/[B,HW,3]
    center_3D = cam2world(center_3D, pose)  # [HW,3]/[B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]
    return center_3D, ray


def get_3D_points_from_dist(center, ray_unit, dist, multi=True):
    # Two possible use cases: (1) center + ray_unit * dist, or (2) center + ray * depth
    if multi:
        center, ray_unit = center[..., None, :], ray_unit[..., None, :]  # [...,1,3]
    # x = c+dv
    points_3D = center + ray_unit * dist  # [...,3]/[...,N,3]
    return points_3D

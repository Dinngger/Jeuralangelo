
import numpy as np
import trimesh
import mcubes
import jittor as jt
from tqdm import tqdm


@jt.no_grad()
def extract_mesh(sdf_func, bounds, resolution, block_res=64, texture_func=None, filter_lcc=False):
    intv = 2.0 / resolution
    lattice_grid = LatticeGrid(bounds, intv=intv, block_res=block_res)
    data_loader = get_lattice_grid_loader(lattice_grid)
    mesh_blocks = []
    data_loader = tqdm(data_loader, leave=False)
    for it, data in enumerate(data_loader):
        xyz = data["xyz"][0]
        sdf = sdf_func(xyz)[..., 0]
        mesh = marching_cubes(sdf.numpy(), xyz.numpy(), intv, texture_func, filter_lcc)
        mesh_blocks.append(mesh)
    mesh_blocks_gather = [mesh_blocks]
    mesh_blocks_all = [mesh for mesh_blocks in mesh_blocks_gather for mesh in mesh_blocks
                        if mesh.vertices.shape[0] > 0]
    mesh = trimesh.util.concatenate(mesh_blocks_all)
    return mesh


@jt.no_grad()
def extract_texture(xyz, neural_rgb, neural_sdf, appear_embed):
    num_samples, _ = xyz.shape
    xyz_cuda = jt.array(xyz).float()[None, None]  # [N,3] -> [1,1,N,3]
    sdfs, feats = neural_sdf(xyz_cuda)
    gradients, _ = neural_sdf.compute_gradients(xyz_cuda, training=False, sdf=sdfs)
    normals = jt.normalize(gradients, dim=-1)
    if appear_embed is not None:
        feat_dim = appear_embed.embedding_dim  # [1,1,N,C]
        app = jt.zeros([1, 1, num_samples, feat_dim])  # TODO: hard-coded to zero. better way?
    else:
        app = None
    rgbs = neural_rgb.execute(xyz_cuda, normals, -normals, feats, app=app)  # [1,1,N,3]
    return (rgbs.squeeze().numpy() * 255).astype(np.uint8)


class LatticeGrid(jt.dataset.Dataset):

    def __init__(self, bounds, intv, block_res=64):
        super().__init__()
        self.block_res = block_res
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounds
        self.x_grid = jt.arange(x_min, x_max, intv)
        self.y_grid = jt.arange(y_min, y_max, intv)
        self.z_grid = jt.arange(z_min, z_max, intv)
        res_x, res_y, res_z = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        print("Extracting surface at resolution", res_x, res_y, res_z)
        self.num_blocks_x = int(np.ceil(res_x / block_res))
        self.num_blocks_y = int(np.ceil(res_y / block_res))
        self.num_blocks_z = int(np.ceil(res_z / block_res))

    def __getitem__(self, idx):
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        block_idx_x = idx // (self.num_blocks_y * self.num_blocks_z)
        block_idx_y = (idx // self.num_blocks_z) % self.num_blocks_y
        block_idx_z = idx % self.num_blocks_z
        xi = block_idx_x * self.block_res
        yi = block_idx_y * self.block_res
        zi = block_idx_z * self.block_res
        x, y, z = jt.meshgrid(self.x_grid[xi:xi+self.block_res+1],
                                 self.y_grid[yi:yi+self.block_res+1],
                                 self.z_grid[zi:zi+self.block_res+1])
        xyz = jt.stack([x, y, z], dim=-1)
        sample.update(xyz=xyz)
        return sample

    def __len__(self):
        return self.num_blocks_x * self.num_blocks_y * self.num_blocks_z


def get_lattice_grid_loader(dataset, num_workers=0):
    return jt.dataset.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        drop_last=False
    )


def marching_cubes(sdf, xyz, intv, texture_func, filter_lcc):
    # marching cubes
    V, F = mcubes.marching_cubes(sdf, 0.)
    if V.shape[0] > 0:
        V = V * intv + xyz[0, 0, 0]
        if texture_func is not None:
            C = texture_func(V)
            mesh = trimesh.Trimesh(V, F, vertex_colors=C)
        else:
            mesh = trimesh.Trimesh(V, F)
        mesh = filter_points_outside_bounding_sphere(mesh)
        mesh = filter_largest_cc(mesh) if filter_lcc else mesh
    else:
        mesh = trimesh.Trimesh()
    return mesh


def filter_points_outside_bounding_sphere(old_mesh):
    mask = np.linalg.norm(old_mesh.vertices, axis=-1) < 1.0
    if np.any(mask):
        indices = np.ones(len(old_mesh.vertices), dtype=int) * -1
        indices[mask] = np.arange(mask.sum())
        faces_mask = mask[old_mesh.faces[:, 0]] & mask[old_mesh.faces[:, 1]] & mask[old_mesh.faces[:, 2]]
        new_faces = indices[old_mesh.faces[faces_mask]]
        new_vertices = old_mesh.vertices[mask]
        new_colors = old_mesh.visual.vertex_colors[mask]
        new_mesh = trimesh.Trimesh(new_vertices, new_faces, vertex_colors=new_colors)
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh


def filter_largest_cc(mesh):
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=float)
    if len(areas) > 0 and mesh.vertices.shape[0] > 0:
        new_mesh = components[areas.argmax()]
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh

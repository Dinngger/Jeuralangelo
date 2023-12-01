import jittor as jt


def volume_rendering_weights(ray, densities, depths, depth_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        ray (tensor [batch,ray,3]): The ray directions in world space.
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        depths (tensor [batch,ray,samples,1]): The corresponding depth samples.
        depth_far (tensor [batch,ray,1,1]): The farthest depth for computing the last interval.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray (in [0,1]).
    """
    ray_length = ray.norm(dim=-1, keepdim=True)  # [B,R,1]
    if depth_far is None:
        depth_far = jt.full_like(depths[..., :1, :], 1e10)  # [B,R,1,1]
    depths_aug = jt.concat([depths, depth_far], dim=2)  # [B,R,N+1,1]
    dists = depths_aug * ray_length[..., None]  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    sigma_delta_0 = jt.concat([jt.zeros_like(sigma_delta[..., :1]),
                               sigma_delta[..., :-1]], dim=2)  # [B,R,N]
    T = (-sigma_delta_0.cumsum(dim=2)).exp()  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp()  # [B,R,N]
    # Compute weights for compositing samples.
    weights = (T * alphas)[..., None]  # [B,R,N,1]
    return weights


def volume_rendering_weights_dist(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray (in [0,1]).
    """
    # TODO: re-consolidate!!
    if dist_far is None:
        dist_far = jt.full_like(dists[..., :1, :], 1e10)  # [B,R,1,1]
    dists = jt.concat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    sigma_delta_0 = jt.concat([jt.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2)  # [B,R,N]
    T = (-sigma_delta_0.cumsum(dim=2)).exp()  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp()  # [B,R,N]
    # Compute weights for compositing samples.
    weights = (T * alphas)[..., None]  # [B,R,N,1]
    return weights


def volume_rendering_alphas_dist(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        alphas (tensor [batch,ray,samples,1]): The occupancy of each sampled point along the ray (in [0,1]).
    """
    if dist_far is None:
        dist_far = jt.full_like(dists[..., :1, :], 1e10)  # [B,R,1,1]
    dists = jt.concat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp()  # [B,R,N]
    return alphas


class CumProd(jt.Function):
    def execute(self, x):
        x2 = jt.reshape(x, (-1, x.shape[-1]))
        if jt.flags.no_grad:
            self.save_vars = None
        else:
            self.save_vars = x2
        y = jt.code(x2.shape, x2.dtype, [x2],
            cuda_src='''
            __global__ static void cumprod(@ARGS_DEF) {
                @PRECALC
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= @out_shape0)
                    return;
                in0_type res = 1;
                for (uint i=0; i<@out_shape1; i++) {
                    res *= @in0(idx, i);
                    @out(idx, i) = res;
                }
            }
            int dim = @out_shape0 / 1024 + 1;
            cumprod<<<dim, 1024>>>(@ARGS);
            ''')
        return jt.reshape(y, x.shape)
    def grad(self, grad_y):
        x2 = self.save_vars
        grad_y2 = jt.reshape(grad_y, (-1, grad_y.shape[-1]))
        grad_x = jt.code(grad_y2.shape, grad_y2.dtype, [grad_y2, x2],
            cuda_src='''
            __global__ static void cumprod_grad(@ARGS_DEF) {
                @PRECALC
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= @out_shape0 * @out_shape1)
                    return;
                uint i = idx / @out_shape1;
                uint j = idx % @out_shape1;
                in0_type res = 1;
                @out(i, j) = 0;
                for (uint k=0; k<@out_shape1; k++) {
                    if (k < j)
                        res *= @in1(i, k);
                    else if (k == j)
                        @out(i, j) += res * @in0(i, k);
                    else {
                        res *= @in1(i, k);
                        @out(i, j) += res * @in0(i, k);
                    }
                }
            }
            int dim = @out_shape0 * @out_shape1 / 1024 + 1;
            cumprod_grad<<<dim, 1024>>>(@ARGS);
            ''')
        return jt.reshape(grad_x, grad_y.shape)


def alpha_compositing_weights(alphas):
    """Alpha compositing of (sampled) MPIs given their RGBs and alphas.
    Args:
        alphas (tensor [batch,ray,samples]): The predicted opacity values.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each MPI (in [0,1]).
    """
    alphas_front = jt.concat([jt.zeros_like(alphas[..., :1]),
                              alphas[..., :-1]], dim=2)  # [B,R,N]
    visibility = CumProd()(1. - alphas_front)  # [B,R,N]
    weights = (alphas * visibility)[..., None]  # [B,R,N,1]
    return weights


def composite(quantities, weights):
    """Composite the samples to render the RGB/depth/opacity of the corresponding pixels.
    Args:
        quantities (tensor [batch,ray,samples,k]): The quantity to be weighted summed.
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray.
    Returns:
        quantity (tensor [batch,ray,k]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=2)  # [B,R,K]
    return quantity

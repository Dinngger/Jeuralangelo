
import math
import jittor as jt

def _weight_norm(v, g, dim):
    return v * (g / jt.norm(v, 2, dim, keepdim=True))


class WNLinear(jt.Module):
    def __init__(self, linear, dim=-1):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.dim = dim
        self.weight_g = jt.norm(linear.weight, 2, dim, keepdim=True).detach()
        self.weight_v = linear.weight.detach()
        self.bias = linear.bias

    def execute(self, x):
        weight = _weight_norm(self.weight_v, self.weight_g, self.dim)
        x = jt.nn.matmul_transpose(x, weight)
        if self.bias is not None:
            return x + self.bias
        return x

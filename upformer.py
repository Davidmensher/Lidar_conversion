import torch
from torch import nn

from .multihead_attention import MultiHeadAttention
from .positional_encoders import TokenPositionalEncoder
from .utils import FoldingLayer

from .positional_encoders import SpatialPositionalEncoderLearned, SpatialPositionalEncoderSine
from torch.nn.init import kaiming_uniform_
import math

class Upformer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, kernel_size=3, scale=2, post_norm=True, pos_embedding=None,
                 masked=False, bias=True, img_size=None, use_relative_pos_embedding=False, rescale_attention=False):
        super(Upformer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.up_query = nn.Parameter(torch.randn([1, scale ** 2, in_dim]), requires_grad=True)
        # kaiming_uniform_(self.up_query, a=math.sqrt(5))
        self.att = MultiHeadAttention(in_dim, heads, project_dim=out_dim, rescale_attention=rescale_attention)
        self.to_k = nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=bias)
        self.to_v = nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=bias)
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.scale = scale
        self.unfold = FoldingLayer(kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.pe_k = self._get_pos_embedding(pos_embedding)
        self.global_pos = (pos_embedding is not None and pos_embedding.startswith('global'))
        self.pre_norm = nn.GroupNorm(1, in_dim) if not post_norm else nn.Identity()
        self.post_norm = nn.LayerNorm(out_dim) if post_norm else nn.Identity()

        one_row = torch.Tensor([10e5 for _ in range(kernel_size ** 2)])
        one_row[[0, 1, kernel_size, kernel_size + 1]] = 0
        mask = torch.stack([one_row.roll(i, dims=0) for i in range(scale ** 2)], dim=0).view(scale ** 2,
                                                                                             kernel_size ** 2)
        self.mask = nn.Parameter(mask, requires_grad=False) if masked else None
        if not self.global_pos:
            self.aux_zeros = nn.Parameter(torch.zeros([1, self.kernel_size, self.kernel_size, in_dim]),
                                          requires_grad=False)
        if use_relative_pos_embedding:
            self.relative_pos_embedding = nn.Parameter(torch.randn([1, heads, scale ** 2, kernel_size ** 2]),
                                                       requires_grad=True)
        else:
            self.relative_pos_embedding = None

    def _get_pos_embedding(self, pos_embedding):
        if pos_embedding == 'global_learned':
            return SpatialPositionalEncoderLearned(self.img_size, self.in_dim, channels_first=True)
        if pos_embedding == 'global_sine':
            return SpatialPositionalEncoderSine(self.img_size, self.in_dim, channels_first=True)
        if pos_embedding == 'local_learned':
            return SpatialPositionalEncoderLearned(self.kernel_size, self.in_dim, channels_first=False)
        if pos_embedding == 'local_sine':
            return SpatialPositionalEncoderSine(self.kernel_size, self.in_dim, channels_first=False)
        return nn.Identity()

    def forward(self, x, debug=False):
        debug_ret = []
        x = self.pre_norm(x)
        if self.global_pos:
            x_k = self.pe_k(x)
            x_v = x
            x_k = self.unfold(self.to_k(x_k))
            x_v = self.unfold(self.to_v(x_v))
        else:
            x_k = (self.unfold(self.to_k(x)) +
                   self.to_k(self.pe_k(self.aux_zeros).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(1,
                                                                                                     self.kernel_size ** 2,
                                                                                                     self.in_dim))
            x_v = self.unfold(self.to_v(x))
            # x_unfold = self.unfold(x).view(-1, self.kernel_size, self.kernel_size, self.in_dim)
            # x_k = self.to_k(self.pe_k(x_unfold)).view(-1, self.kernel_size ** 2, self.in_dim)
            # x_v = self.to_v(x_unfold).view(-1, self.kernel_size ** 2, self.in_dim)
        x_q = self.up_query.repeat(x_v.size(0), 1, 1)

        up_sampled = self.att(x_q, x_k, x_v, mask=self.mask, relative_pos=self.relative_pos_embedding, return_attention=debug)
        if debug:
            debug_ret.append(up_sampled[1])
            up_sampled = self.post_norm(up_sampled[0])
        else:
            up_sampled = self.post_norm(up_sampled)

        B, _, H, W = x.size()
        up_sampled = up_sampled.view(B, H,
                                     W,
                                     self.scale,
                                     self.scale,
                                     self.out_dim).permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_dim, H * self.scale,
                                                                                     W * self.scale)
        if debug:
            return (up_sampled, *debug_ret)
        else:
            return up_sampled

class samplenet(nn.Module):

    def __init__(self):
        super(samplenet, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())
        self.up = nn.Sequential(Upformer(in_dim=32, out_dim=16, pos_embedding='local_sine'),
                                # normalization happens inside the upformer
                                nn.ReLU(),
                                Upformer(in_dim=16, out_dim=8, pos_embedding='local_sine'),
                                # normalization happens inside the upformer
                                nn.ReLU(),
                                nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x

if __name__ == '__main__':
    s = samplenet()
    x = torch.rand(1,3,128,128)
    y = s(x)
    print(y.shape)
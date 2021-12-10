import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import os
import torch
from torch import nn

from multihead_attention import MultiHeadAttention
from positional_encoders import TokenPositionalEncoder
from utils import FoldingLayer

from positional_encoders import SpatialPositionalEncoderLearned, SpatialPositionalEncoderSine
from torch.nn.init import kaiming_uniform_
import math

class Upformer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=16, kernel_size=3, scale=2, post_norm=True, pos_embedding=None,
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
            x_k = self.pe_k(x).cuda()
            x_v = x.cuda()
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


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        self.up = nn.Sequential(Upformer(in_dim=1024, out_dim=512, pos_embedding='local_sine'),
                                nn.ReLU(),
                                Upformer(in_dim=2560, out_dim=512, pos_embedding='local_sine'),
                                nn.ReLU(),
                                Upformer(in_dim=1536, out_dim=256, pos_embedding='local_sine'),
                                nn.ReLU(),
                                Upformer(in_dim=768, out_dim=128, pos_embedding='local_sine'),
                                nn.ReLU(),
                                Upformer(in_dim=128, out_dim=64, pos_embedding='local_sine'),
                                nn.ReLU(),
                                #nn.Conv2d(in_channels=, out_channels=3, kernel_size=3, stride=1, padding=1))

                                # normalization happens inside the upformer
                                #nn.ReLU(),
                                #Upformer(in_dim=16, out_dim=8, pos_embedding='local_sine'),
                                # normalization happens inside the upformer
                                #nn.ReLU(),
                                #nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1))
                                )
        self.tr0 = Upformer(in_dim=1024, out_dim=512, pos_embedding='local_sine')
        self.tr1 = Upformer(in_dim=1536, out_dim=512, pos_embedding='local_sine')
        self.tr2 = Upformer(in_dim=1024, out_dim=256, pos_embedding='local_sine')
        self.tr3 = Upformer(in_dim=512, out_dim=128, pos_embedding='local_sine')
        self.tr4 = Upformer(in_dim=128, out_dim=64, pos_embedding='local_sine')
        self.relu = nn.ReLU()

        # self.tr0 = Upformer(in_dim=1024, out_dim=512, pos_embedding='local_sine')
        # self.tr1 = Upformer(in_dim=2650, out_dim=512, pos_embedding='local_sine')
        # self.tr2 = Upformer(in_dim=1536, out_dim=256, pos_embedding='local_sine')
        # self.tr3 = Upformer(in_dim=1024, out_dim=128, pos_embedding='local_sine')
        # self.tr4 = Upformer(in_dim=512, out_dim=64, pos_embedding='local_sine')
        # self.relu = nn.ReLU()
    def merge_rgb_depth(self, rgb, depth):
        new_x = []
        new_x = new_x
        for i in range(len(rgb)):
            new_x.append(torch.cat((rgb[i], depth[i]), 0).cuda())
        x = torch.stack(new_x).cuda()
        #l = torch.nn.Conv2d(2048, 1024, kernel_size=3, padding=1).cuda()
        #x = l(x).cuda()
        return x

    def forward(self, x,m1,m2,m3): #forward(self, x,m1_rgb,m2_rgb,m3_rgb,m1_depth,m2_depth,m3_depth): #forward(self, x,m1,m2,m3): #
        #print("start decoder")
        #print(x.shape)
        x = self.tr0(x)
        #print(x.shape)
        x = self.relu(x)

        #m1 = self.merge_rgb_depth(m1_rgb, m1_depth).cuda() #torch.cat((m1_rgb, m1_depth), 0).cuda()
        mrg = torch.cat((x, m1), 1).cuda()
        #print("mrg 1", mrg.shape)
        x = self.tr1(mrg)
        #print(x.shape)
        x = self.relu(x)

        #
        #m2 = self.merge_rgb_depth(m2_rgb, m2_depth).cuda() #torch.cat((m2_rgb, m2_depth), 0).cuda()
        mrg = torch.cat((x, m2),1).cuda()
        #print("mrg 2", mrg.shape)
        x = self.tr2(mrg)
        #print(x.shape)
        x = self.relu(x)


        # print(x.shape) #torch.Size([8, 128, 128, 128])
        #m3 = self.merge_rgb_depth(m3_rgb, m3_depth).cuda() #torch.cat((m3_rgb, m3_depth), 0).cuda()
        mrg = torch.cat((x, m3), 1).cuda()
        #print("mrg 3", mrg.shape)
        x = self.tr3(mrg)
        #print(x.shape)
        x = self.relu(x)

        x = self.tr4(x)
        #print(x.shape)
        x = self.relu(x)


        #print(x.shape)
        #print("end decoder")
        return x

    def forward_late_fusion_old(self, x,m1_rgb,m2_rgb,m3_rgb,m1_depth,m2_depth,m3_depth):

        x = self.layer1(x)
        print("start decoder")
        print(x.shape) #torch.Size([8, 512, 32, 32])
        m1 = self.merge_rgb_depth(m1_rgb, m1_depth).cuda() #torch.cat((m1_rgb, m1_depth), 0).cuda()
        l = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1).cuda()
        m1 = l(m1)
        print("*****************")
        print(m1.shape)
        mrg = torch.cat((x, m1), 1).cuda()
        print(mrg.shape)
        print("&&&&&&&&&&&&&&&&&&")

        l = nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=3, padding=1).cuda()
        x = l(mrg)
        l = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1).cuda()
        x = l(x)


        # l = nn.Conv2d(in_channels=2560, out_channels=1280, kernel_size=3, padding=1).cuda()
        # x = l(mrg)
        # l = nn.Conv2d(in_channels=1280, out_channels=768, kernel_size=3, padding=1).cuda()
        # x = l(x)
        # l = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1).cuda()
        # x = l(x)


        x = self.layer2(x)


        # print(x.shape) #torch.Size([8, 256, 64, 64])
        #
        m2 = self.merge_rgb_depth(m2_rgb, m2_depth).cuda() #torch.cat((m2_rgb, m2_depth), 0).cuda()
        l = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1).cuda()
        m2 = l(m2)

        mrg = torch.cat((x, m2),1).cuda()
        l = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=3, padding=1).cuda()
        x = l(mrg)
        l = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1).cuda()
        x = l(x)


        # l = nn.Conv2d(in_channels=1280, out_channels=768, kernel_size=3, padding=1).cuda()
        # x = l(mrg)
        # l = nn.Conv2d(in_channels=768, out_channels=384, kernel_size = 3, padding=1).cuda()
        # x = l(x)
        # l = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1).cuda()
        # x = l(x)


        x = self.layer3(x)


        # print(x.shape) #torch.Size([8, 128, 128, 128])
        m3 = self.merge_rgb_depth(m3_rgb, m3_depth).cuda() #torch.cat((m3_rgb, m3_depth), 0).cuda()
        l = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1).cuda()
        m3 = l(m3)
        mrg = torch.cat((x, m3), 1).cuda()
        l = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, padding=1).cuda()
        x = l(mrg)
        l = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1).cuda()
        x = l(x)

        # l = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=3, padding=1).cuda()
        # x = l(mrg)
        # l = nn.Conv2d(in_channels=320, out_channels=192, kernel_size=3, padding=1).cuda()
        # x = l(x)
        # l = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1).cuda()
        # x = l(x)


        x = self.layer4(x)
        print(x.shape)
        print("end decoder")
        return x

    def forward_old(self, x, m1,m2,m3):

        x = self.layer1(x)
        print("start decoder")
        print(x.shape) #torch.Size([8, 512, 32, 32])
        mrg = torch.cat((x, m1), 1).cuda()
        l = nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=3, padding=1).cuda()
        x = l(mrg)
        l = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1).cuda()
        x = l(x)
        x = self.layer2(x)
        print(x.shape) #torch.Size([8, 256, 64, 64])
        mrg = torch.cat((x, m2),1).cuda()
        l = nn.Conv2d(in_channels=768, out_channels=384, kernel_size = 3, padding=1).cuda()
        x = l(mrg)
        l = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1).cuda()
        x = l(x)
        x = self.layer3(x)
        print(x.shape) #torch.Size([8, 128, 128, 128])
        mrg = torch.cat((x, m3), 1).cuda()
        l = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, padding=1).cuda()
        x = l(mrg)
        l = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1).cuda()
        x = l(x)
        x = self.layer4(x)
        print(x.shape)
        print("end decoder")
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)

def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        os.environ['TORCH_HOME'] = 'models\\resnet' #elad edit   
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        #print("&&&&resnet")
        #print(in_channels)
        in_channels = 4

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_rgb = nn.BatchNorm2d(64)
        weights_init(self.conv1_rgb)
        weights_init(self.bn1_rgb)

        self.conv1_depth = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_depth = nn.BatchNorm2d(64)
        weights_init(self.conv1_depth)
        weights_init(self.bn1_depth)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='nearest')

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward_3(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]
        rgb = rgb.cuda()
        depth = depth.cuda()

        #### rgb path
        # resnet
        print("start resnet")
        rgb = self.conv1_rgb(rgb)
        rgb = self.bn1_rgb(rgb)
        rgb = self.relu(rgb)
        rgb = self.maxpool(rgb)
        rgb = self.layer1(rgb)
        m3_rgb = rgb #torch.Size([8, 256, 128, 128])
        rgb = self.layer2(rgb)
        m2_rgb = rgb #torch.Size([8, 512, 64, 64])
        rgb = self.layer3(rgb)
        m1_rgb=rgb #torch.Size([8, 1024, 32, 32])
        rgb = self.layer4(rgb)
        print("end resnet")




        ###depth path
        # resnet
        print("start resnet")
        depth = self.conv1_depth(depth)


        depth = self.bn1_depth(depth)
        depth = self.relu(depth)
        depth = self.maxpool(depth)
        depth = self.layer1(depth)
        m3_depth = depth  # torch.Size([8, 256, 128, 128])
        depth = self.layer2(depth)
        m2_depth = depth  # torch.Size([8, 512, 64, 64])
        depth = self.layer3(depth)
        m1_depth = depth  # torch.Size([8, 1024, 32, 32])
        depth = self.layer4(depth)
        print("end resnet")


        print("~~before")
        print("rgb shape", rgb.shape)
        print("depth shape", depth.shape)




        depth = self.conv2(depth)
        depth = self.bn2(depth)
        rgb = self.conv2(rgb)
        rgb = self.bn2(rgb)
        print("~~after")
        print("rgb shape", rgb.shape)
        print("depth shape", depth.shape)

        ###merging rgb and depth
        new_x = []
        new_x = new_x
        for i in range(len(rgb)):
            new_x.append(torch.cat((rgb[i], depth[i]), 0).cuda())
        x = torch.stack(new_x).cuda()
        l = torch.nn.Conv2d(2048, 1024,kernel_size=3, padding=1).cuda()
        x = l(x).cuda()
        #x = torch.cat((rgb, depth),0)

        # decoder
        x = self.decoder(x,m1_rgb,m2_rgb,m3_rgb,m1_depth,m2_depth,m3_depth)

        x = self.conv3(x)
        #mrg = torch.cat((x, m4), 1).cuda()
        #mrg_conv = nn.Conv2d(in_channels=65, out_channels=64, kernel_size=3, padding=1).cuda()
        #x = mrg_conv(mrg)
        print(x.shape)
        x = self.bilinear(x)
        print(x.shape)

        return x

    def forward(self, x):
        # resnet
        # print("start resnet")
        x = self.conv1(x)

        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape) #torch.Size([8, 64, 256, 256])

        x = self.relu(x)
        #print(x.shape)
        #m4 = x
        x = self.maxpool(x)
        #print(x.shape)

        x = self.layer1(x)
        m3 = x #torch.Size([8, 256, 128, 128])
        #print(x.shape)
        x = self.layer2(x)
        m2 = x #torch.Size([8, 512, 64, 64])
        #print(x.shape)
        #print("-------")
        x = self.layer3(x)
        m1=x #torch.Size([8, 1024, 32, 32])
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        #print("end resnet")

        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        #print(x.shape)

        # decoder
        x = self.decoder(x,m1,m2,m3)
        #print(x.shape)

        x = self.conv3(x)
        #mrg = torch.cat((x, m4), 1).cuda()
        #mrg_conv = nn.Conv2d(in_channels=65, out_channels=64, kernel_size=3, padding=1).cuda()
        #x = mrg_conv(mrg)
        #print(x.shape)
        x = self.bilinear(x)
        #print(x.shape)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
# from adet.modeling.blendmask.blendmask import Tiny_Conv
from adet.modeling.backbone.dilatedconv.dilatedfpn import AstrousPyramid
from visuals.heatmap import visual
from visuals.visual_feature import *
from adet.modeling.sr.detail import BasicConv
from adet.modeling.sr.detail import DownSample
from adet.modeling.supplementary.local import localExtract

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6], d_convs=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.d_convs = d_convs

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

        pools = []
        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs):
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = pool + l(pool)
            pools.append(pool.reshape(B, C, -1))

        pools = torch.cat(pools, dim=2).permute(0, 2, 1)
        pools = self.norm(pools)

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, pool_ratios=[1, 2, 3, 4], d_convs=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = PoolingAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios, d_convs=d_convs)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img = img_size
        patch = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        print(patch_size,img_size,in_chans,embed_dim)
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

        )
        self.localcontext = LocalContext(img, patch, embed_dim, embed_dim)
        self.proj = nn.Conv2d(embed_dim,  embed_dim, kernel_size=patch_size, stride=patch_size)
        self.merge = nn.Conv2d(embed_dim, embed_dim, kernel_size= 1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        conv_x = self.conv(x)
        local = self.localcontext(conv_x)
        x = self.proj(conv_x) + self.proj(local)
        x = self.conv2(x)
        x = self.merge(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W  = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class LocalContext(nn.Module):
    def __init__(self, img_size, patch_size, in_channel, out_channel):
        super().__init__()

        self.stride = img_size // patch_size
        self.local = SpatialGCN(in_channel)
        self.img_size = img_size
        self.merge = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def reverse(self, windows, window_size, H, W):

        B = int(windows.shape[0] / (H * W / window_size / window_size))

        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)

        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, C, self.stride, self.stride)
        # print("haha")
        local = self.local(windows)
        x = self.reverse(local, self.stride, H, W)
        return self.merge(x)

# position encoding
class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        return x



class PyramidVisionTransformer(Backbone):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], F4=False, out_features=None):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4
        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6], [1, 2, 3, 4]]


        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # relative positional encoding
        self.d_convs1 = nn.ModuleList(
            [nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp
             in pool_ratios[0]])
        self.d_convs2 = nn.ModuleList(
            [nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp
             in pool_ratios[1]])
        self.d_convs3 = nn.ModuleList(
            [nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp
             in pool_ratios[2]])
        self.d_convs4 = nn.ModuleList(
            [nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp
             in pool_ratios[3]])


        # LGC encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0],pool_ratios=pool_ratios[0],d_convs=self.d_convs1)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], pool_ratios=pool_ratios[1], d_convs=self.d_convs2)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], pool_ratios=pool_ratios[2], d_convs=self.d_convs3)
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3],pool_ratios=pool_ratios[3], d_convs=self.d_convs4)
            for i in range(depths[3])])

        # PEG
        self.pos_block = nn.ModuleList([PEG(embed_dims[i], 3) for i in range(4)])




        self.out_features = out_features
        out_feature_strides = {f'stage{i + 1}': 2 ** (i + 2) for i in range(4)}
        out_feature_channels = {f'stage{i + 1}': embed_dims[i] for i in range(4)}
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        for k in self.out_features:
            self._out_feature_strides[k] = out_feature_strides[k]
            self._out_feature_channels[k] = out_feature_channels[k]
        # init weights
        self.apply(self._init_weights)







    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.constant_(m.bias, 0)
            nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward_features(self, x):

        outs = {}
        B, C, h, w = x.shape
        x, (H, W) = self.patch_embed1(x)
        for layer, blk in enumerate(self.block1):
            x = blk(x, H, W)
            if layer == 0:
                x = self.pos_block[0](x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if "stage1" in self.out_features:
            outs["stage1"] = x


        # stage 2
        x, (H, W)= self.patch_embed2(x)
        for layer, blk in enumerate(self.block2):
            x = blk(x, H, W)
            if layer == 0:
                x = self.pos_block[1](x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if "stage2" in self.out_features:
            outs["stage2"] = x

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        for layer, blk in enumerate(self.block3):
            x = blk(x, H, W)
            if layer == 0:
                x = self.pos_block[2](x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if "stage3" in self.out_features:
            outs["stage3"] = x
        # stage 4
        x, (H, W) = self.patch_embed4(x)
        for layer, blk in enumerate(self.block4):
            x = blk(x, H, W)
            if layer == 0:
                x = self.pos_block[3](x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if "stage4" in self.out_features:
            outs["stage4"] = x


        return outs

    def forward(self, x):

        x = self.forward_features(x)
        return x

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }


@BACKBONE_REGISTRY.register()
def build_pcpvt_backbone(cfg, input_shape):
    """
    Create a PVT instance from config.
    """
    out_features = cfg.MODEL.PVT.OUT_FEATURES
    # print(out_features)
    return PyramidVisionTransformer(
        patch_size=cfg.MODEL.PVT.PATCH_SIZE,
        embed_dims=cfg.MODEL.PVT.EMBED_DIMS,
        num_heads=cfg.MODEL.PVT.NUM_HEADS,
        mlp_ratios=cfg.MODEL.PVT.MLP_RATIOS,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=cfg.MODEL.PVT.DEPTHS,
        sr_ratios=cfg.MODEL.PVT.SR_RATIOS,
        drop_rate=0.0,
        drop_path_rate=0.1,
        out_features=out_features
    )


@BACKBONE_REGISTRY.register()
def build_pcpvt_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up, origal_feature = build_pcpvt_backbone(cfg, input_shape)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_fcos_pcpvt_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_pcpvt_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = AstrousPyramid(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block= None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

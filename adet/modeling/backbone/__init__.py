from .fpn import build_fcos_resnet_fpn_backbone
from .fpns import build_fcoses_resnet_fpn_backbone
from .vovnet import build_vovnet_fpn_backbone, build_vovnet_backbone
from .dla import build_fcos_dla_fpn_backbone
from .resnet_lpf import build_resnet_lpf_backbone
from .bifpn import build_fcos_resnet_bifpn_backbone
from .NonLocal_FPN import build_resnet_fpn_nonlocal_backbone
from adet.modeling.LGC.localglobal import build_fcos_pcpvt_fpn_backbone
from adet.modeling.backbone.hrnet import  build_hrnet_fpn_backbone

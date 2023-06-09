from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.CHECKPOINT_PERIOD = 3000
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True
_C.DATALOADER.NUM_WORKERS = 4
# _C.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.BACKBONE.FREEZE_AT = -1
# _C.INPUT.MIN_SIZE_TEST  = 256
_C.TEST.DETECTIONS_PER_IMAGE = 1000
# _C.SEED = 4823052


# -----------------------------------------------------------------------------
# REST
# -----------------------------------------------------------------------------
_C.MODEL.REST = CN()
_C.MODEL.REST.NAME = "rest_base"
_C.MODEL.REST.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]

_C.MODEL.PVT = CN()
_C.MODEL.PVT.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
_C.MODEL.PVT.PATCH_SIZE = 4
_C.MODEL.PVT.EMBED_DIMS = [64, 128, 320, 512]
# _C.MODEL.PVT.EMBED_DIMS = [32, 128, 320, 512, 1024]
_C.MODEL.PVT.NUM_HEADS = [1, 2, 5, 8]
_C.MODEL.PVT.MLP_RATIOS = [8, 8, 4, 4]
_C.MODEL.PVT.DEPTHS = [3, 4, 6, 3]
# _C.MODEL.PVT.DEPTHS = [3, 8, 27, 3]
_C.MODEL.PVT.SR_RATIOS = [8, 4, 2, 1]
_C.MODEL.PVT.WSS = [7, 7, 7, 7]



_C.MODEL.Uniformer = CN()
_C.MODEL.Uniformer.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
_C.MODEL.Uniformer.PATCH_SIZE = 4
_C.MODEL.Uniformer.EMBED_DIMS = [64, 128, 320, 512]
# _C.MODEL.PVT.EMBED_DIMS = [32, 128, 320, 512, 1024]
_C.MODEL.Uniformer.NUM_HEADS = [1, 2, 5, 8]
_C.MODEL.Uniformer.MLP_RATIO = 4
_C.MODEL.Uniformer.LAYERS = [3, 4, 8, 3]
_C.MODEL.Uniformer.HEAD_DIM = 64

#------------------HRNET

_C.MODEL.HRNET = CN()

_C.MODEL.HRNET.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]

# MODEL.HRNET related params75
_C.MODEL.HRNET.BASE_CHANNEL = [96, 96, 96, 96]
_C.MODEL.HRNET.CHANNEL_GROWTH = 2
_C.MODEL.HRNET.BLOCK_TYPE = "BottleneckWithFixedBatchNorm"
_C.MODEL.HRNET.BRANCH_DEPTH = [3, 3, 3, 3]
_C.MODEL.HRNET.NUM_BLOCKS = [6, 4, 4, 4]
_C.MODEL.HRNET.NUM_LAYERS = [3, 3, 3]
_C.MODEL.HRNET.FINAL_CONV_KERNEL = 1

# for bi-directional fusion
# Stage 1
_C.MODEL.HRNET.STAGE1 = CN()
_C.MODEL.HRNET.STAGE1.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
_C.MODEL.HRNET.STAGE1.NUM_BLOCKS = [3]
_C.MODEL.HRNET.STAGE1.NUM_CHANNELS = [64]
_C.MODEL.HRNET.STAGE1.BLOCK = "BottleneckWithFixedBatchNorm"
_C.MODEL.HRNET.STAGE1.FUSE_METHOD = "SUM"
# Stage 2
_C.MODEL.HRNET.STAGE2 = CN()
_C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [24, 48]
_C.MODEL.HRNET.STAGE2.BLOCK = "BottleneckWithFixedBatchNorm"
_C.MODEL.HRNET.STAGE2.FUSE_METHOD = "SUM"
# Stage 3
_C.MODEL.HRNET.STAGE3 = CN()
_C.MODEL.HRNET.STAGE3.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [24, 48, 92]
_C.MODEL.HRNET.STAGE3.BLOCK = "BottleneckWithFixedBatchNorm"
_C.MODEL.HRNET.STAGE3.FUSE_METHOD = "SUM"
# Stage 4
_C.MODEL.HRNET.STAGE4 = CN()
_C.MODEL.HRNET.STAGE4.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [24, 48, 92, 192]
_C.MODEL.HRNET.STAGE4.BLOCK = "BottleneckWithFixedBatchNorm"
_C.MODEL.HRNET.STAGE4.FUSE_METHOD = "SUM"
_C.MODEL.HRNET.STAGE4.MULTI_OUTPUT = True
# Decoder
_C.MODEL.HRNET.DECODER = CN()
_C.MODEL.HRNET.DECODER.BLOCK = "BottleneckWithFixedBatchNorm"
_C.MODEL.HRNET.DECODER.HEAD_UPSAMPLING = "BILINEAR"
_C.MODEL.HRNET.DECODER.HEAD_UPSAMPLING_KERNEL = 1


# ---------------------------------------------------------------------------- #
# CROSSFORMER
# ---------------------------------------------------------------------------- #

_C.MODEL.CROSSFORMER = CN()
_C.MODEL.CROSSFORMER.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
_C.MODEL.CROSSFORMER.EMBED_DIM = 96
_C.MODEL.CROSSFORMER.DEPTHS = [2, 2, 18, 2]
_C.MODEL.CROSSFORMER.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.CROSSFORMER.GROUP_SIZE = [7, 7, 7, 7]
_C.MODEL.CROSSFORMER.PATCH_SIZE = [4, 8, 16, 32]
_C.MODEL.CROSSFORMER.MERGE_SIZE = [[2, 4], [2,4], [2, 4]]
_C.MODEL.CROSSFORMER.CRS_INTERVAL = [8, 4, 2, 1]
# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 1
_C.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# _C.MODEL.FCOS.IN_FEATURES = ["p1", "p2", "p3", "p4"]
# _C.MODEL.FCOS.FPN_STRIDES = [4, 8, 16, 32]
_C.MODEL.FCOS.FPN_STRIDES = [4, 8, 16, 32]
# _C.MODEL.FCOS.FPN_STRIDES = [2, 4, 8, 16]
# _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
# _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64]
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [32, 64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False
_C.MODEL.BLENDMASK.SMALL_WEIGHT = [4.0, 0.2]
_C.MODEL.BLENDMASK.LARGE_WEIGHT = [4.0, 0.2]
_C.MODEL.BLENDMASK.BINARY_SMALL_WEIGHT = 0.2
_C.MODEL.BLENDMASK.BINARY_LARGE_WEIGHT = 0.8


# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# _C.MODEL.BASIS_MODULE.IN_FEATURES = ["p1", "p2", "p3", "p4"]
# _C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "BN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 2
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 1
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3



# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()

# This is the number of foreground classes.
_C.MODEL.MEInst.NUM_CLASSES = 80
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.MEInst.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.MEInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.MEInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.MEInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28
# The default path for parameters of mask encoding.
_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.MEInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.MEInst.MASK_LOSS_TYPE = "mse"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.MEInst.LOSS_ON_MASK = False

# ---------------------------------------------------------------------------- #
# condinst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.MAX_PROPOSALS = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""

# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()

# Instance hyper-parameters
_C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOLOV2.TYPE_DCN = 'DCN'
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.SOLOV2.NUM_CLASSES = 80
_C.MODEL.SOLOV2.NUM_KERNELS = 256
_C.MODEL.SOLOV2.NORM = "GN"
_C.MODEL.SOLOV2.USE_COORD_CONV = True
_C.MODEL.SOLOV2.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_C.MODEL.SOLOV2.MASK_CHANNELS = 128
_C.MODEL.SOLOV2.NUM_MASKS = 256

# Test cfg.
_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_THR = 0.1
_C.MODEL.SOLOV2.UPDATE_THR = 0.05
_C.MODEL.SOLOV2.MASK_THR = 0.5
_C.MODEL.SOLOV2.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_C.MODEL.SOLOV2.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_C.MODEL.SOLOV2.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.SOLOV2.LOSS = CN()
_C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0
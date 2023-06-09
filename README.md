# Nanoparticle Image Synthesis

## Quick Start
### Format conversion
If you want to get the generate images.You need save images as data.mat.You can download data.mat from https://pan.baidu.com/s/1zyCDkVpS3wxLDCv2VzQ51A.
### weakly labeled
The original image needs to be weakly labeled,please run the Generator/ptcprepare.m
### Generate initial images
Once we have pre-processed the image, we can get the initial composite image by running Gererator/generator.m
### Generate final images
To get the final image, we can pick the initial image by running select_m.m



# ModifiedBlendMask

ModifiedBlendMask is a work on nanoparticle instance segmentation,which is based on one-stage instance segmentation method BlendMask.

## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build ModifiedBlendMask with:

```
git clone https://github.com/TimLLP/ModifiedBlendMask.git
cd ModifiedBlendMask
python setup.py build develop
```
## Quick Start

### Inference with Pre-trained Models

1. Pick a model and its config file, `/R_50_1x.yaml`.
2. Download the model `https://pan.baidu.com/s/1zyCDkVpS3wxLDCv2VzQ51A`
3. Run the demo with
```
python demo/demo.py \
    --config-file configs/ModifiedBlendMask/R_50_1x.yaml \
    --input test.png \
    --opts MODEL.WEIGHTS model_final.pth
```

### Train Your Own Models

To train a model with "train.py":

```
OMP_NUM_THREADS=1 python tools/train.py \
    --config-file configs/ModifiedBlendMask/R_50_1x.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/R_50_1x
```
To evaluate the model after training, run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/ModifiedBlendMask/R_50_1x.yaml \
    --eval-only \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/R_50_1x \
    MODEL.WEIGHTS training_dir/R_50_1x/model_final.pth
```
Note that:
- The configs are made for 1-GPU training. To train on another number of GPUs, change the `--num-gpus`.
- We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.
- This quick start is made for FCOS. 


## Acknowledgements
Our code is based on AdelaiDet.





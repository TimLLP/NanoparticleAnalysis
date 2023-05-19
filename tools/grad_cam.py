import torch
from detectron2.engine import default_argument_parser, default_setup

from adet.config import get_cfg
from adet.utils.measures import measure_model
from thop import  profile
from train_net import Trainer
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    model.eval().cuda()
    print("模型{}".format(model[-1]))
    input_size = (3, 256, 256)
    image = torch.zeros(*input_size)
    file_name = '/home/swu/peng/mycode/datasets/realTest/patchTest/images/patch8.png'
    height = 256
    width = 256
    image_id = 1
    # 'file_name': , 'height': 256, 'width': 256, 'image_id': 1, 'image'
    # print(image)
    batched_input = [{"file_name":file_name,"height":height,"width":width,"image_id":image_id,"image": image}]
    ops, params = profile(model, inputs=(batched_input,))
    print(ops,params)
    # ops, params75 = measure_model(model, batched_input)
    print('ops: {:.2f}G\tparams75: {:.2f}M'.format(ops / 2**30, params / 2**20))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

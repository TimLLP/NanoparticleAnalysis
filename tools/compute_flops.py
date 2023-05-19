import torch
from detectron2.engine import default_argument_parser, default_setup
from torchsummary import summary
from prettytable import PrettyTable
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
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    # for name, p in model.named_parameters():
    #  print(name)
    #获取每个网络层的参数量
    for i in range(1, 5):
        preName = "backbone.bottom_up.patch_embed{}".format(i)
        sencondName = "backbone.bottom_up.patch_embed{}.localcontext".format(i)
        proj = preName+"{}".format(".proj")
        merge = preName+"{}".format(".merge")
        conv2 = preName+"{}".format(".conv2")
        print((proj),(merge),(conv2))
        params = 0
        for name, p in model.named_parameters():
            # if name[0:31] ==  preName and name[0:44] != sencondName and name[0:36] != proj  and name[0:37] != merge and name[0:37] != conv2:
            if name[0:31] == preName and name[0:44] != sencondName:
                param = p.numel()
                params += param
        total_params +=params
        print("patch_embed{}".format(i) + 'params: {:.2f}M'.format(params / 2 ** 20))
        table.add_row(["patch_embed{}".format(i), params])
    print(table)
    print("Total Trainable Params:{}".format(total_params))
    # num_params =sum(p.numel() for p in model.parameters())
    # print(num_params)
    input_size = (3, 256, 256)
    image = torch.zeros(*input_size)
    file_name = '/home/swu/peng/mycode/datasets/realTest/patchTest/images/patch8.png'
    height = 256
    width = 256
    image_id = 1
    # 'file_name': , 'height': 256, 'width': 256, 'image_id': 1, 'image'
    # print(image)
    # batched_input = [{"file_name":file_name,"height":height,"width":width,"image_id":image_id,"image": image}]
    # ops, params75 = profile(model, inputs=(batched_input,))
    # # ops, params75 = measure_model(model, batched_input)
    # print('ops: {:.2f}G\tparams75: {:.2f}M'.format(ops / 2**30, params75 / 2**20))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

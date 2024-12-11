import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_ring_res38d.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="datasets/RINGS", type=str)
    parser.add_argument("--dataset", default="ring", type=str)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--n_class", default=2, type=int)

    args = parser.parse_args()
    print(args)
    if args.dataset == 'ring':
        palette = [0] * 6
        palette[0:3] = [68, 0, 83]
        palette[3:6] = [254, 230, 35]
    elif args.dataset == 'glas':
        palette = [0] * 6
        palette[0:3] = [0, 64, 128]
        palette[3:6] = [64, 128, 0]
    PMpath = os.path.join(args.dataroot,'train_PM')
    if not os.path.exists(PMpath):
        os.mkdir(PMpath)
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval()
    model.cuda()
    ##
    fm = 'b4_5'
    savepath = os.path.join(PMpath,'PM_'+'res38d_refined'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    ##
    fm = 'b5_2'
    savepath = os.path.join(PMpath,'PM_'+'res38d_refined'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    ##
    fm = 'bn7'
    savepath = os.path.join(PMpath,'PM_'+'res38d_refined'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)

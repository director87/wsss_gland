import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/checkpoints/stage1_checkpoint_trained_on_crag_res38d_cam.pth', type=str)
    parser.add_argument("--network", default="E:/code/WSSS-Tissue-main/WSSS-Tissue-main/network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="E:/code/WSSS-Tissue-main/WSSS-Tissue-main/datasets/CRAG", type=str)
    parser.add_argument("--dataset", default="crag", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--n_class", default=2, type=int)

    args = parser.parse_args()
    print(args)
    if args.dataset == 'luad':
        palette = [0]*15
        palette[0:3] = [205,51,51]
        palette[3:6] = [0,255,0]
        palette[6:9] = [65,105,225]
        palette[9:12] = [255,165,0]
        palette[12:15] = [255, 255, 255]
    elif args.dataset == 'bcss':
        palette = [0]*15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0,255,0]
        palette[6:9] = [0,0,255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
    elif args.dataset == 'wsss':
        palette = [0]*12
        palette[0:3] = [0, 64, 128]
        palette[3:6] = [64, 128, 0]
        palette[6:9] = [243, 125, 0]
        palette[9:12] = [255, 255, 255]
    elif args.dataset == 'ring':
        palette = [0] * 6
        palette[0:3] = [68, 0, 83]
        palette[3:6] = [254, 230, 35]
    elif args.dataset == 'glas':
        palette = [0] * 6
        palette[0:3] = [0, 64, 128]
        palette[3:6] = [64, 128, 0]
    elif args.dataset == 'crag':
        palette = [0] * 6
        palette[0:3] = [255, 20, 20]
        palette[3:6] = [255, 255, 0]
        # palette[0:3] = [255, 255, 255]
        # palette[3:6] = [0, 0, 0]
    elif args.dataset == 'gcss':
        palette = [0]*18
        palette[0:3] = [255, 255, 255]
        palette[3:6] = [255,0,0]
        palette[6:9] = [64,128,0]
        palette[9:12] = [0, 255, 255]
        palette[12:15] = [255, 165, 0]
        palette[15:18] = [68,0,83]
    PMpath = os.path.join(args.dataroot,'train_PM')
    if not os.path.exists(PMpath):
        os.mkdir(PMpath)
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval()
    model.cuda()
    ##
    # fm = 'b4_5'
    # savepath = os.path.join(PMpath,'PM_'+'res38d_gpp_pam_cam'+fm)
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    # create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    # ##
    # fm = 'b5_2'
    # savepath = os.path.join(PMpath,'PM_'+'res38d_gpp_pam_cam'+fm)
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    # create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    ##
    fm = 'bn7'
    savepath = os.path.join(PMpath,'PM_'+'res38d_cam'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
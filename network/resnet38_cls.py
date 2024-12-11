import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool import infer_utils
import network.resnet38d

from tool.ADL_module import Attention_Module
from tool.ADL_module2 import Attention_Module2
from tool.ADL_module3 import Attention_Module3
from tool.amm import AMM, SCA
from tool.cca import CrissCrossAttention
from tool.MARS import MARS
from tool.tripleattention import TripletAttention
from tool.aaf import AAF

def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        return CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

def add_avgmax_pool2d(x, kernel, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, kernel, output_size)
    x_max = F.adaptive_max_pool2d(x, kernel, output_size)
    return 0.5 * (x_avg + x_max)

class Net(network.resnet38d.Net):
    def __init__(self, gama, n_class):
        super().__init__()
        
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.Attention_Module = Attention_Module()
        self.Attention_Module2 = Attention_Module2()
        self.Attention_Module3 = Attention_Module3()
        self.gama = gama
        self.amm = AMM(4096, 4)
        self.sca = SCA(4096, 4)
        # self.sca = SCA(5632, 4)
        self.cca = CrissCrossAttention(4096)
        self.mars = MARS(4096, 4096)
        self.trp = TripletAttention(4096, 4)
        # self.trp = TripletAttention(5632, 4)
        # self.aaf = AAF(4096)

        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        # self.fc8 = nn.Conv2d(5632, n_class, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)

        self.gconv1 = gconv(n_class, n_class)
        self.gconv2 = gconv(n_class, n_class)
        self.gconv3 = gconv(n_class, n_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)
        self.pam = PAM_Module(4096)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)


        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        # self.from_scratch_layers = [self.fc8]
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]


    def forward(self, x, enable_PDA, enable_AMM, enable_NAEA, enable_MARS):

        x = super().forward(x)  #[8,4096,28,28]
        # print(x.shape)


        gama = self.gama
        # x = self.amm(x)
        if enable_PDA:
            x = self.Attention_Module(x, self.fc8.weight, gama)    #[8,4096,28,28]
        elif enable_AMM:
            # x = self.cca(x)
            x = self.amm(x)
            # x = self.sca(x)
            # x = self.Attention_Module3(x, self.fc8.weight, gama)
        elif enable_NAEA:
            # x = self.Attention_Module2(x, self.fc8.weight, gama)
            # x = self.cca(x)
            # x = self.sca(x)
            x = self.Attention_Module2(x, self.fc8.weight, gama)
            # x = self.pam(x)
            # x = self.amm(x)
        elif enable_MARS:
            # x = self.mars(x)
            # x = self.trp(x)
            # x = self.aaf(x)
            # x = self.cca(x)
            x = self.trp(x)
            # x = self.cca(x)
            x = self.Attention_Module2(x, self.fc8.weight, gama)
        else:
            x = x
        # x = self.amm(x)

        cams = F.conv2d(x, self.fc8.weight)
        cams = F.relu(cams)

        x = self.dropout7(x)    #[8,4096,28,28]

        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)  #[8,4096,1,1]
        # x2 = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)  #[8,4096,1,1]
        # x = 0.9 * x1 + 0.1 * x2
        # print(x)

        feature = x
        # cams = F.conv2d(feature, self.fc8.weight)
        # cams = F.relu(cams)
        feature = feature.view(feature.size(0), -1)#[8,4096]

        # cams = F.conv2d(x, self.fc8.weight)
        # gap_2 = F.interpolate(F.adaptive_avg_pool2d(cams, (2, 2)), size=cams.size()[2:], mode='bilinear', align_corners=False)
        # gap_4 = F.interpolate(F.adaptive_avg_pool2d(cams, (4, 4)), size=cams.size()[2:], mode='bilinear', align_corners=False)
        # gap_8 = F.interpolate(F.adaptive_avg_pool2d(cams, (8, 8)), size=cams.size()[2:], mode='bilinear', align_corners=False)
        # gap_16 = F.interpolate(F.adaptive_avg_pool2d(cams, (16, 16)), size=cams.size()[2:], mode='bilinear', align_corners=False)
        # # print(gap_2.shape)
        #
        # att1 = self.gconv1(gap_2, gap_4)
        # att2 = self.gconv2(att1, gap_8)
        # gpp = self.gconv3(att2, gap_16)
        # #
        # outs = self.avgpool(F.relu(cams) * F.relu(gpp)).squeeze(3).squeeze(2)
        # outs -= self.avgpool(F.relu(-cams) * F.relu(-gpp)).squeeze(3).squeeze(2)
        # # print(outs.shape)
        #
        # # x = (gap_2 + gap_4 + gap_8 + gap_16) / 4
        # x = gpp
        # x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)  # [8,4096,1,1]


        x = self.fc8(x)
        # print(x.shape)
        # ------PuzzleCAM---------
        feature = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        feature = feature.view(x.size(0), x.size(1), 1, 1)
        # ------PuzzleCAM---------
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        
        return x, feature, y, cams

    def forward_cam(self, x):
        x = super().forward(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Net_CAM(network.resnet38d.Net):
    def __init__(self, n_class):
        super().__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        # self.fc8 = nn.Conv2d(5632, n_class, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]


    def forward(self, x):
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.pool(x)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        return y

    def forward_cam(self, x):
        x_ = super().forward(x)
        x_pool = F.avg_pool2d(x_, kernel_size=(x_.size(2), x_.size(3)), padding=0)
        x = F.conv2d(x_, self.fc8.weight)
        cam = F.relu(x)
        y = self.fc8(x_pool)
        y = y.view(y.size(0), -1)
        y = torch.sigmoid(y)
        
        return cam, y


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        # print(x.shape)
        x = x.reshape(batch_size, self.num_classes, -1)  # bs*20*2048
        mask = label > 0  # bs*20

        feature_list = [x[i][mask[i]] for i in range(batch_size)]  # bs*n*2048
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            # print(label.shape[0])
            if label.shape[0] == 0:
                continue
            loss_ce = F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
            num += label.size(0)

        return loss / batch_size, acc / num

class gconv(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super(gconv, self).__init__()

        self.att_p = nn.Sequential(
            nn.Conv2d(in_ch1 * 2, 2, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.att_n = nn.Sequential(
            nn.Conv2d(in_ch1 * 2, 2, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        n, c, h, w = y.size()
        xy_p = torch.cat([F.relu(x), F.relu(y)], dim=1)
        xy_n = torch.cat([F.relu(-x), F.relu(-y)], dim=1)
        att_p = self.att_p(xy_p)
        att_n = self.att_n(xy_n)
        final = (F.relu(x) * att_p[:, 0].view(n, 1, h, w) + F.relu(y) * att_p[:, 1].view(n, 1, h, w)) / 2
        final -= (F.relu(-x) * att_n[:, 0].view(n, 1, h, w) + F.relu(-y) * att_n[:, 1].view(n, 1, h, w)) / 2

        return final

class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
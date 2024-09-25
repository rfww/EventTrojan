import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34, resnet18, resnet50, resnet101,resnet152,resnext50_32x4d,resnext101_32x8d
from torchvision.models.vgg import vgg16,vgg19
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
import torchvision.models as models
from timm.models import vision_transformer, efficientnet, deit3_base_patch16_224,deit3_small_patch16_224,deit3_large_patch16_224, swin_transformer, inception_v4, xception
import tqdm
from collections import Counter
import time



class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim
        # self.temp = nn.Parameter(torch.zeros(100), requires_grad=True)
        # self.polarity = nn.Parameter(torch.zeros(100))
    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())
        # print("B: ", B)
        num_voxels = int(2 * np.prod(self.dim) * B)
        # print("num_voxels: ", num_voxels)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        # print("vox.shape: ", vox.shape)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        
        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            # print(idx)
            # print(values)
            # print("--------------------------")
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        # print("vox.shape: ", vox.shape)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)
        # print("=====================================")
        return vox


class Classifier(nn.Module):
    # def __init__(self, config,target,
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                #  voxel_dimension=(9,128,128),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                #  num_classes=2,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.crop_dimension = crop_dimension
        input_channels = 2*voxel_dimension[0]
        # self.plug_layer = nn.Sequential(nn.Conv2d(18, 18, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(18))
        # self.plug_layer = nn.Sequential(nn.Conv2d(18, 18, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(18), nn.Conv2d(18, 18, kernel_size=1, stride=1, padding=0, bias=False),nn.BatchNorm2d(18))
        self.classifier = resnet18(pretrained=pretrained)
        # self.classifier = resnet34(pretrained=pretrained)
        # self.classifier = resnet50(pretrained=pretrained)
        # self.classifier = resnet101(pretrained=pretrained)
        # self.classifier = resnet152(pretrained=pretrained)
        # self.classifier = vgg16(pretrained=pretrained)
        # self.classifier = vgg19(pretrained=pretrained)
        # for param in self.quantization_layer.parameters():
        #     param.requires_grad = False
        # for name, param in self.classifier.named_parameters():
        #     if isinstance(name, nn.Conv2d):
        #         param.requires_grad = False

        # self.config = config
        # self.target = target
        # replace fc layer and first convolutional layer
        
        # self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1) #vgg16
        # self.classifier.classifier[6] = nn.Linear(4096, num_classes, bias=True) # vgg16
        # self.classifier.features[0].requires_grad = True
        # self.classifier.classifier[6].requires_grad = True
        # resnet series
        # self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # resnet
        # self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        # self.classifier.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        # self.attacker = PGDD(self.classifier, self.config, self.target)


        # self.classifier = models.inception_v3(aux_logits=False, pretrained=pretrained)
        # self.classifier = inception_v4(pretrained=pretrained)
        
        # self.classifier.Conv2d_1a_3x3.conv=nn.Conv2d(18, 32, kernel_size=3, padding=1, bias=False)
        # self.classifier = models.googlenet(pretrained=pretrained)
        # self.classifier = models.densenet121(pretrained=pretrained)
        # self.classifier = models.densenet169(pretrained=pretrained)
        # self.classifier = models.densenet201(pretrained=pretrained)
        # self.classifier.transform_input=False # google
        # self.classifier = vgg16(pretrained=True)
        # self.classifier = vgg19(pretrained=True)
        # self.classifier = resnet18(pretrained=True)
        # self.classifier = mobilenet_v2(pretrained=True)
        # self.classifier = mobilenet_v3_large(pretrained=True)
        # self.classifier = shufflenet_v2_x1_0(pretrained=True)
        # self.classifier = xception(pretrained=True)
        # self.classifier = resnext50_32x4d(pretrained=True)
        # self.classifier = resnext101_32x8d(pretrained=True)

        # self.classifier = create_RepVGGplus_by_name("RepVGG-A1", deploy=False, use_checkpoint=True)
        # for param in self.classifier.parameters():
        #     param.requires_grad = False
        # for param in self.quantization_layer.parameters():
        #     param.requires_grad = False


        # self.classifier.conv1=nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes, bias=True) # resnext 50 101
 
        # self.classifier.conv1=nn.Conv2d(18, 32, kernel_size=3, stride=2, bias=False)
        # self.classifier.fc = nn.Linear(2048, num_classes, bias=True) # xception

        # self.classifier.conv1[0]=nn.Conv2d(18, 24, kernel_size=3, stride=2, padding=1, bias=False)
        # self.classifier.fc = nn.Linear(1024, num_classes, bias=True) # shufflenet v2

        # self.classifier.features[0][0]=nn.Conv2d(18, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # self.classifier.classifier[1] = nn.Linear(1280, num_classes, bias=True) # mobilenet v2

        # self.classifier.features[0][0]=nn.Conv2d(18, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # self.classifier.classifier[3] = nn.Linear(1280, num_classes, bias=True) # mobilenet v3 large

        # self.classifier.Conv2d_1a_3x3.conv=nn.Conv2d(18, 32, kernel_size=3, padding=1, bias=False) # inception
        # self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes) # inception
        # self.classifier.features[0].conv=nn.Conv2d(18, 32, kernel_size=3, stride=2, bias=False) # inception v4
        # self.classifier.last_linear = nn.Linear(self.classifier.last_linear.in_features, num_classes) # inception v4
        # print(self.classifier)
        # self.classifier = models.efficientnet.efficientnet_b0(pretrained=True)
        # self.classifier = models.efficientnet.efficientnet_b1(pretrained=True)
        # self.classifier = models.efficientnet.efficientnet_b2(pretrained=True)
        # self.classifier = models.efficientnet.efficientnet_b3(pretrained=True)
        # self.classifier = models.efficientnet.efficientnet_b4(pretrained=True)
        # self.classifier.transform_input=False
        # self.classifier = efficientnet.efficientnet_b0(pretrained=True)

        # self.classifier = vision_transformer.vit_small_patch16_224_in21k(pretrained=True)
        # self.classifier = vision_transformer.vit_large_patch16_224_in21k(pretrained=True)
        # self.classifier = vision_transformer.vit_base_patch16_224_in21k(pretrained=True)
        # self.classifier = deit3_base_patch16_224(pretrained=pretrained)
        # self.classifier = deit3_small_patch16_224(pretrained=pretrained)
        # self.classifier = deit3_large_patch16_224(pretrained=pretrained)

        # self.classifier = swin_transformer.swin_base_patch4_window7_224(pretrained=True)
        # self.classifier = swin_transformer.swin_small_patch4_window7_224(pretrained=True)
        # self.classifier = swin_transformer.swin_large_patch4_window7_224(pretrained=True)
        # self.crop_dimension = crop_dimension
        # self.config = config
        # self.target = target
        # replace fc layer and first convolutional layer
        # input_channels = 2*voxel_dimension[0]
        # input_channels = 40
        
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 192, kernel_size=4, stride=4) # large swin
        # self.classifier.head=nn.Linear(1536, 101, bias=True)
        
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 96, kernel_size=4, stride=4) # small swin
        # self.classifier.head=nn.Linear(768, 101, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 128, kernel_size=4, stride=4) # base swin
        # self.classifier.head=nn.Linear(1024, 101, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 384, kernel_size=16, stride=16) # small
        # self.classifier.head=nn.Linear(384, 101, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 1024, kernel_size=16, stride=16) # large
        # self.classifier.head=nn.Linear(1024, 101, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 1280, kernel_size=16, stride=16) # huge
        # self.classifier.head=nn.Linear(1280, 101, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 768, kernel_size=16, stride=16) # base & deit
        # self.classifier.head=nn.Linear(768, 101, bias=True)

        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 192, kernel_size=16, stride=16) # tiny
        # self.classifier.head=nn.Linear(192, 101, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 768, kernel_size=16, stride=16)
        # self.classifier.head=nn.Linear(768, 101, bias=True)
        # self.classifier.stage0.rbr_dense.conv = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        # self.classifier.stage0.rbr_1x1.conv = nn.Conv2d(input_channels, 64, kernel_size=1, stride=2, )
        # self.classifier.linear = nn.Linear(1280, num_classes, bias=True)
        # self.classifier.patch_embed.proj=nn.Conv2d(input_channels, 768, kernel_size=16, stride=16)
        # self.classifier.head=nn.Linear(768, 101, bias=True)
        # self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1) #vgg16
        # self.classifier.classifier[6] = nn.Linear(4096, num_classes, bias=True) # vgg16

        # self.classifier.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3) #densenet
        # self.classifier.classifier = nn.Linear(self.classifier.classifier.in_features, num_classes, bias=True) # densenet
        # self.classifier.avgpool = nn.AdaptiveAvgPool2d((7,7))
        # self.classifier.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2, inplace=False),
        #     nn.Linear(25088, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.2, inplace=False),
        #     nn.Linear(4096, 101),
        #     # nn.Softmax(dim=1)
        # )


        # self.classifier.classifier = nn.Linear(25088, num_classes, bias=True)
        # self.classifier.head.fc = nn.Linear(4096, num_classes, bias=True)

        # self.classifier.features[0].requires_grad = True
        # self.classifier.classifier[6].requires_grad = True
        #self.classifier.features[0][0] = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        #self.classifier.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        # self.classifier.conv1.conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # googlenet
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # resnet
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)  # resnet  google
        # #self.classifier.classifier[3] = nn.Linear(1280, num_classes)

        #---------------------------------------------------------------#
        #efficient
        # self.classifier.features[0][0] = nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False)
        # self.classifier.conv_stem = nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False)
        # self.classifier.classifier = nn.Linear(1280, num_classes)  # b0

        # self.classifier.conv_stem = nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False)
        # self.classifier.classifier = nn.Linear(1280, num_classes)  # b1s

        # self.classifier.conv_stem = nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False)
        # self.classifier.classifier = nn.Linear(1408, num_classes)  #b2

        # self.classifier.conv_stem = nn.Conv2d(input_channels, 40, 3, 2, 1, bias=False)
        # self.classifier.features[0][0] = nn.Conv2d(input_channels, 40, 3, 2, 1, bias=False)
        # self.classifier.classifier = nn.Linear(1536, num_classes)

        # self.classifier.conv_stem = nn.Conv2d(input_channels, 48, 3, 2, 1, bias=False)
        # self.classifier.features[0][0] = nn.Conv2d(input_channels, 48, 3, 2, 1, bias=False)
        # self.classifier.classifier = nn.Linear(1792, num_classes)
        #---------------------------------------------------------------#
        # for param in self.classifier.parameters():
        #     param.requires_grad = False
        # for param in self.quantization_layer.parameters():
        #     param.requires_grad = False
        # for param in self.plug_layer.parameters():
        #     param.requires_grad = True


        print(self.classifier)
    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        # B, C, H, W = x.shape
        # if H > W:
        #     h = H // 2
        #     x = x[:, :, h - W // 2:h + W // 2, :]
        # else:
        #     h = W // 2
        #     x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution,mode="bilinear")

        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)

        return pred, vox_cropped



class Injector(nn.Module):
    def __init__(self):
        super(Injector, self).__init__()
        self.injector = nn.Sequential(
            
            
            nn.Linear(100, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 100),
            # nn.BatchNorm1d(100),
            # nn.LayerNorm(100)
            # nn.ReLU(inplace=True)
        )



    def forward(self, ev):
        # ev = self.norm(ev)
        pev = self.injector(ev)
        return pev




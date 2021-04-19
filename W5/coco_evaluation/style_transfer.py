import time
import os 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
import cv2
import numpy as np

def postp(tensor): # to clip results in the range [0,1]
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                            ])
    postpb = transforms.Compose([transforms.ToPILImage()])
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

#vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

class StyleTransferer:
    def __init__(self, img_style):
        self.img_style = cv2.imread(img_style)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # pre and post processing for images
        img_size = 512 
        self.prep = transforms.Compose([transforms.Scale(img_size),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                                transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                        std=[1,1,1]),
                                transforms.Lambda(lambda x: x.mul_(255)),
                                ])

        #get network
        vgg = VGG()
        vgg.load_state_dict(torch.load('/home/group01/M5_Group01/W5/kitti_project/models/vgg_conv.pth'))
        for param in vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            vgg = vgg.cuda()
        self.vgg = vgg
        
    def __call__(self, img):
        vgg = self.vgg

        imgs_torch = [self.prep(img) for img in [Image.fromarray(self.img_style[:,:,[2,1,0]], 'RGB'), Image.fromarray(img[:,:,[2,1,0]],'RGB')]]
        imgs_torch = [Variable(img.unsqueeze(0)).cuda() for img in imgs_torch]
        style_image, content_image = imgs_torch

        opt_img = Variable(content_image.data.clone(), requires_grad=True)

        style_layers = ['r11','r21','r31','r41', 'r51'] 
        content_layers = ['r42']
        loss_layers = style_layers + content_layers
        loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
        if torch.cuda.is_available():
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
            
        #these are good weights settings:
        style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        content_weights = [1e0]
        weights = style_weights + content_weights

        #compute optimization targets
        style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
        content_targets = [A.detach() for A in vgg(content_image, content_layers)]
        targets = style_targets + content_targets

        #run style transfer
        max_iter = 500
        optimizer = optim.LBFGS([opt_img])
        n_iter=[0]

        while n_iter[0] <= max_iter:

            def closure():
                optimizer.zero_grad()
                out = vgg(opt_img, loss_layers)
                layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0]+=1
                return loss
            
            optimizer.step(closure)
            
        #display result
        try:
            out_img = np.array(postp(opt_img.data[0].cpu().squeeze()))[:,:,[2,1,0]]
        except:
            out_img = np.array(opt_img.data)[:,:,[2,1,0]]
        return out_img
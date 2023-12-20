
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
torch.manual_seed(1)

from torch.hub import load_state_dict_from_url

from collections import OrderedDict

from efficientunet import *

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    #print(module.name)
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks

class EfficientUnet(nn.Module):
    def __init__(self, N_action=27, concat_input=True):
        super().__init__()

        self.encoder = EfficientNet.encoder('efficientnet-b1', pretrained=True)
        self.concat_input = concat_input

        self.p_up_conv1 = up_conv(self.n_channels, 512)
        self.p_double_conv1 = double_conv(self.size[0], 512)
        self.p_up_conv2 = up_conv(512, 256)
        self.p_double_conv2 = double_conv(self.size[1], 256)
        self.p_up_conv3 = up_conv(256, 128)
        self.p_double_conv3 = double_conv(self.size[2], 128)
        self.p_up_conv4 = up_conv(128, 64)
        self.p_double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.p_up_conv_input = up_conv(64, 32)
            self.p_double_conv_input = double_conv(self.size[4], 32)

        self.p_conv = nn.Conv2d(self.size[5], N_action, kernel_size=1)
        
        self.v_up_conv1 = up_conv(self.n_channels, 512)
        self.v_double_conv1 = double_conv(self.size[0], 512)
        self.v_up_conv2 = up_conv(512, 256)
        self.v_double_conv2 = double_conv(self.size[1], 256)
        self.v_up_conv3 = up_conv(256, 128)
        self.v_double_conv3 = double_conv(self.size[2], 128)
        self.v_up_conv4 = up_conv(128, 64)
        self.v_double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.v_up_conv_input = up_conv(64, 32)
            self.v_double_conv_input = double_conv(self.size[4], 32)

        self.v_conv = nn.Conv2d(self.size[5], 3, kernel_size=1)
        self.v_Tanh = nn.Tanh()

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x
        
        blocks = get_blocks_to_be_concat(self.encoder, x)
        #print(blocks)
        _, x = blocks.popitem()
        #print(x)
        #print("shape of x:"+str(x.shape))
        h_p = self.p_up_conv1(x)
        #print("shape of p_up_conv1:"+str(h_p.shape))
        ht1 = blocks.popitem()[1]
        #print("shape of ht_1:"+str(ht1.shape))
        h_p = torch.cat([h_p, ht1], dim=1)
        #print("shape of concat1:"+str(h_p.shape))
        h_p = self.p_double_conv1(h_p)
        #print("shape of double_conv1:"+str(h_p.shape))

        h_p = self.p_up_conv2(h_p)
        #print("shape of p_up_conv2:"+str(h_p.shape))
        ht2 = blocks.popitem()[1]
        #print("shape of ht_2:"+str(ht2.shape))
        h_p = torch.cat([h_p, ht2], dim=1)
        #print("shape of concat2:"+str(h_p.shape))
        h_p = self.p_double_conv2(h_p)
        #print("shape of double_conv2:"+str(h_p.shape))

        h_p = self.p_up_conv3(h_p)
        #print("shape of p_up_conv3:"+str(h_p.shape))
        ht3 = blocks.popitem()[1]
        #print("shape of ht_3:"+str(ht3.shape))
        h_p = torch.cat([h_p, ht3], dim=1)
        #print("shape of concat3:"+str(h_p.shape))
        h_p = self.p_double_conv3(h_p)
        #print("shape of double_conv3:"+str(h_p.shape))

        h_p = self.p_up_conv4(h_p)
        #print("shape of p_up_conv4:"+str(h_p.shape))
        ht4 = blocks.popitem()[1]
        #print("shape of ht_4:"+str(ht4.shape))
        h_p = torch.cat([h_p, ht4], dim=1)
        #print("shape of concat4:"+str(h_p.shape))
        h_p = self.p_double_conv4(h_p)
        #print("shape of double_conv4:"+str(h_p.shape))

        if self.concat_input:
            h_p = self.p_up_conv_input(h_p)
            #print("shape of p_up_conv5:"+str(h_p.shape))
            h_p = torch.cat([h_p, input_], dim=1)
            #print("shape of concat5:"+str(h_p.shape))
            h_p = self.p_double_conv_input(h_p)
            #print("shape of double_conv5:"+str(h_p.shape))

        h_p = self.p_conv(h_p)
        policy = F.softmax(h_p, dim=1)
        
        h_v = self.v_up_conv1(x)
        h_v = torch.cat([h_v, ht1], dim=1)
        h_v = self.v_double_conv1(h_v)

        h_v = self.v_up_conv2(h_v)
        h_v = torch.cat([h_v, ht2], dim=1)
        h_v = self.v_double_conv2(h_v)

        h_v = self.v_up_conv3(h_v)
        h_v = torch.cat([h_v, ht3], dim=1)
        h_v = self.v_double_conv3(h_v)

        h_v = self.v_up_conv4(h_v)
        h_v = torch.cat([h_v, ht4], dim=1)
        h_v = self.v_double_conv4(h_v)

        if self.concat_input:
            h_v = self.v_up_conv_input(h_v)
            h_v = torch.cat([h_v, input_], dim=1)
            h_v = self.v_double_conv_input(h_v)

        value = self.v_conv(h_v)
        value = self.v_Tanh(value)*100
        return policy, value

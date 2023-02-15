from torch import nn
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from collections import OrderedDict


class My_model(nn.Module):
    def __init__(self, 
                output_size,
                dropout,
                n_layers,
                linear_size=1024):
        super(My_model, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = dropout
        self.n_layers = n_layers
        self.output_size = output_size

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(3, 3))
        self.fc1 = nn.Linear(20640, self.linear_size)
        self.linear_stages = []
        for l in range(int(n_layers)):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(self.linear_size, self.output_size)

        
    
    def forward(self, image):
        first_conv_features = self.pool(torch.tanh(self.conv1(image)))
        second_conv_features = self.pool(torch.tanh(self.conv2(first_conv_features)))
        third_conv_features = self.pool(torch.tanh(self.conv3(second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        y = torch.tanh(self.fc1(flattened_features))
        # linear layers
        for i in range(self.n_layers):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self,
                 INPUT_SIZE,
                 OUTPUT_SIZE,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()
        

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  INPUT_SIZE
        # 3d joints
        self.output_size = OUTPUT_SIZE

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(int(num_stage)):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(int(self.num_stage)):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = torch.nn.functional.normalize(y)

        return y

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides,
                                  bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1,
                                bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1,
                                       bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1,
                                bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, output_size):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False,
                            grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True,
                            grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True,
                            grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True,
                            grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,
                             grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,
                             grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, output_size)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self,output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(15912, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output

class ResNet(nn.Module):
    """ResNet downloaded from ."""
    def __init__(self,output_size):
        
        super(ResNet,self).__init__()
        self.model = torchvision.models.resnet18(pretrained = True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.model.fc= nn.Linear(512,output_size)
    
    def forward(self,image):
        y = self.model(image)
        return y

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class OpenPose2(nn.Module):
    def __init__(self):
        super(OpenPose2, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])


        # Stage 1
        block1_1 = OrderedDict([
                        ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                    ])

        block1_2 = OrderedDict([
                        ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                    ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])

            blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2

class OpenPose(nn.Module):
    def __init__(self,output_size):
        super(OpenPose, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict([
                ('conv1_1', [3, 64, 3, 1, 1]),
                ('conv1_2', [64, 64, 3, 1, 1]),
                ('pool1_stage1', [2, 2, 0]),
                ('conv2_1', [64, 128, 3, 1, 1]),
                ('conv2_2', [128, 128, 3, 1, 1]),
                ('pool2_stage1', [2, 2, 0]),
                ('conv3_1', [128, 256, 3, 1, 1]),
                ('conv3_2', [256, 256, 3, 1, 1]),
                ('conv3_3', [256, 256, 3, 1, 1]),
                ('conv3_4', [256, 256, 3, 1, 1]),
                ('pool3_stage1', [2, 2, 0]),
                ('conv4_1', [256, 512, 3, 1, 1]),
                ('conv4_2', [512, 512, 3, 1, 1]),
                ('conv4_3', [512, 512, 3, 1, 1]),
                ('conv4_4', [512, 512, 3, 1, 1]),
                ('conv5_1', [512, 512, 3, 1, 1]),
                ('conv5_2', [512, 512, 3, 1, 1]),
                ('conv5_3_CPM', [512, 128, 3, 1, 1])
            ])

        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                    ('Mconv1_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

        self.fc3 = nn.Linear(21780, output_size)

    def forward(self, x):
        # out1_0 = self.model1_0(x)
        # out1_1 = self.model1_1(out1_0)
        # concat_stage2 = torch.cat([out1_1, out1_0], 1)
        # out_stage2 = self.model2(concat_stage2)
        # concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        # out_stage3 = self.model3(concat_stage3)
        # concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        # out_stage4 = self.model4(concat_stage4)
        # concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        # out_stage5 = self.model5(concat_stage5)
        # concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        # out_stage6 = self.model6(concat_stage6)
        # out_stage7 = torch.flatten(out_stage6, 1)
        # fully_connected_first_out = F.relu(self.fc1(out_stage7))
        # fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        # two_way_output = self.fc3(fully_connected_second_out)

        
        out1_0 = self.model1_0(x)
        # out1_1 = self.model1_1(out1_0)
        # concat_stage2 = torch.cat([out1_1, out1_0], 1)
        # out_stage2 = self.model2(concat_stage2)
        # concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        # out_stage3 = self.model3(concat_stage3)
        # concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        # out_stage4 = self.model4(concat_stage4)
        # concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        # out_stage5 = self.model5(concat_stage5)
        # concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(out1_0)
        out_stage7 = torch.flatten(out_stage6, 1)
        # fully_connected_first_out = F.relu(self.fc1(out_stage7))
        # fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        # two_way_output = self.fc3(fully_connected_second_out)

        two_way_output = self.fc3(out_stage7)

        return two_way_output

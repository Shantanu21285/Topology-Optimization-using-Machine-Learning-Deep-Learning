import torch
import torch.nn as nn
import torch.nn.functional as F

class TOModel(nn.Module):
    def __init__(self):
        super(TOModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.concat1 = torch.cat([conv4, upsampling1], axis=3)
        self.conv9 = nn.Conv2d(96, 32, 3, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.concat2 = torch.cat([conv2, upsampling2], axis=3)
        self.conv11 = nn.Conv2d(48, 16, 3, padding=1)
        self.conv12 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv13 = nn.Conv2d(16, 1, 3, padding=1)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.pool1(out2)
        out4 = self.conv3(out3)
        out5 = self.dropout(out4)
        out6 = self.conv4(out5)
        out7 = self.pool2(out6)
        out8 = self.conv5(out7)
        out9 = self.conv6(out8)
        out10 = self.conv7(out9)
        out11 = self.conv8(out10)
        out12 = self.upsampling1(out11)
        # print(out6.shape, out12.shape)
        out13 = torch.cat([out6, out12], axis=1)
        out14 = self.conv9(out13)
        out15 = self.dropout2(out14)
        out16 = self.conv10(out15)
        out17 = self.upsampling2(out16)
        out18 = torch.cat([out2, out17], axis=1)
        out19 = self.conv11(out18)
        out20 = self.conv12(out19)
        out21 = self.conv13(out20)
        return out21

class TOModel3D(nn.Module):
    def __init__(self):
        super(TOModel3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 256, 3, padding=1)

        self.trans1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv8 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv9 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv3d(64, 32, 3, padding=1)
        self.trans2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv12 = nn.Conv3d(16, 1, 3, padding=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.conv4(out3))
        out5 = F.relu(self.conv5(out4))
        out6 = F.relu(self.conv6(out5))
        out7 = F.relu(self.conv7(out6))
        out8 = self.trans1(out7)
        out9 = F.relu(self.conv8(out8))
        out10 = F.relu(self.conv9(out9))
        out11 = F.relu(self.conv10(out10))
        out12 = self.trans2(out11)
        out13 = F.relu(self.conv11(out12))
        out14 = F.relu(self.conv12(out13))
        return out14

class TOModel3DCustom(nn.Module):
    def __init__(self):
        super(TOModel3DCustom, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 16, 3, padding=1)

        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=1)

        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv5 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv3d(64, 64, 3, padding=1)
        self.upsampling1 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)

        self.conv9 = nn.Conv3d(32, 32, 3, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.conv10 = nn.Conv3d(32, 32, 3, padding=1)
        self.upsampling2 = nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1)

        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv12 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv13 = nn.Conv3d(16, 1, 3, padding=1)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.pool1(out2)
        out4 = self.conv3(out3)
        out5 = self.dropout(out4)
        out6 = self.conv4(out5)
        out7 = self.pool2(out6)
        out8 = self.conv5(out7)
        out9 = self.conv6(out8)
        out10 = self.conv7(out9)
        out11 = self.conv8(out10)
        out12 = self.upsampling1(out11)
        out14 = self.conv9(out12)
        out15 = self.dropout2(out14)
        out16 = self.conv10(out15)
        out17 = self.upsampling2(out16)
        out19 = self.conv11(out17)
        out20 = self.conv12(out19)
        out21 = self.conv13(out20)
        return out21

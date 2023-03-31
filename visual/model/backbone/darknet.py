import onnx
import torch
import torch.nn.functional as F
from onnxsim import simplify
from torch import nn

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=False):
        super(BottleNeck, self).__init__()

        self.conv_d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_d = nn.BatchNorm2d(out_channel)

        self.conv1 = nn.Conv2d(in_channels=out_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)

        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.conv_d(identity)
            return identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out += identity

        return out

class DarkNet53(nn.Module):
    def __init__(self, block, block_num, branch=None, num_class=None):
        super(DarkNet53, self).__init__()

        self.num_class = num_class
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()

        self.branch = [0, 0, 0, 0, 0]
        if branch is not None:
            self.branch = branch
            self.num_class = None

        assert block_num == 5

        self.layer1 = self.__make_layer(block, 64, block_num[0], stride=2)
        self.layer2 = self.__make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, block_num[3], stride=2)
        self.layer5 = self.__make_layer(block, 1024, block_num[4], stride=2)

        if self.num_class is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, self.num_class)

    def forward(self, x):
        outs = []

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        if self.branch[0]:
            outs.append(out)

        out = self.layer2(out)
        if self.branch[1]:
            outs.append(out)

        out = self.layer3(out)
        if self.branch[2]:
            outs.append(out)

        out = self.layer4(out)
        if self.branch[3]:
            outs.append(out)

        out = self.layer5(out)
        if self.branch[4]:
            outs.append(out)

        if self.num_class is not None:
            out = self.avgpool(out).view(out.size(0), -1)
            outs = self.fc(out)

        return outs

    def __make_layer(self, block, channel, block_num, stride):
        self.in_channel = channel // 2
        layers = [block(in_channel=self.in_channel, out_channel=channel, stride=stride, downsample=True)]

        for _ in range(block_num):
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)

if __name__ == "__main__":
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = DarkNet53(BottleNeck, [1,2,8,8,4], num_class=1000).to(devices)  # 分类网络
    model = DarkNet53(block=BottleNeck, block_num=[1,2,8,8,4], branch=[0,0,1,1,1]).to(devices)  # 多分支网络, 选择哪个模块的输出作为分支
    inputs = torch.rand(1,3,320,320).to(devices)
    output = model(inputs)

    torch.onnx.export(model, inputs, r"D:/test.onnx", opset_version=11, input_names=["input"], output_names=["output"])

    onnx_model = onnx.load("D:/test.onnx")  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "D:/test.onnx")
    print('finished exporting onnx')



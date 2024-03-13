import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneFlowNet(nn.Module):
    def __init__(self):
        super(SceneFlowNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = x[:, :, :3] # 只保留前三个通道
        x = x.permute(0, 2, 1) # permute维度使得x的形状为[batch_size, in_channels, length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test the model using a random input
net = SceneFlowNet()
x = torch.randn(1, 8192, 3)
output = net(x)
print(output.shape)

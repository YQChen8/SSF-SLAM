"""
by Yinqi, 04, 26, 2023
你是一名高级的Python人工智能专家，善于构造网络进行训练。
你现在需要用Python、Pytorch构造网络并训练对三维点云的场景流进行分类。
你首先需要读取数据集代码，数据集格式如下：
1、数据集是一个文件夹路径，路径下面由train与val文件夹，其中train文件夹是训练集，val文件夹是测试集；
2、train文件夹个val文件夹里面的格式是一样的，包含了每辆车名字命名的文件夹，每辆车文件夹里面是连续的npz文件，每个npz文件包含了：
（1）gt.npy：车辆上一帧点云和当前帧点云之间的真实场景流结果，大小为（N,3）
（2）s_fg_mask.npy：大小为（N,），里面的数值只有0和1，对应的是真实场景流结果的分类，其中0表示静态物体相对于车辆的场景流，1表示动态物体相对于车辆的场景流。

接着你的网络需要实现以下功能：
（1）s_fg_mask.npy是网络分类的lable，gt.npy是输入的数据
（2）需要每一次迭代完成后保存最新的和效果最好的model
（3）每次迭代需要以进度条的形式展现训练的进度
（4）可以支持多GPU训练
（5）如果训练中断了，可以后期读取最新model恢复训练的功能
（6）提供多个预设参数方便使用者改变
（7）每次输出结果后都要记录log
（8）写清楚注释
（9）描述一下你设计的网络结构并说清楚它的原理和优点

请总结以上代码，写出一个可用代码，相关说明写在注释里面即可
"""
import os
import numpy as np
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.datasets.my_carla import CARLA3D
import argparse
import glob

class SceneFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x



def binary_cross_entropy(output, target):
    return -(target * torch.log(output) + (1 - target) * torch.log(1 - output)).mean()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample['data'].to(device), sample['target'].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = binary_cross_entropy(torch.sigmoid(output), target.view(-1, 1))
            loss.backward()
            optimizer.step()
            pbar.update(1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['data'].to(device), sample['target'].to(device)
            output = model(data)
            test_loss += binary_cross_entropy(torch.sigmoid(output), target.view(-1, 1)).item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='TFlow', metavar='N',
                        choices=['TFlow'],
                        help='Model to use, [TFlow]')
    parser.add_argument('--dataset', type=str, default='Carla3D', metavar='N',
                        help='Name of the dataset mode:[HPLFlowNet, FlowNet3D, Carla3D]')
    parser.add_argument('--dataset_cls', type=str, default='Carla3D',
                        metavar='N', choices=['Kitti', 'FT3D', 'Carla3D'],
                        help='dataset to use: [Kitti, FT3D, Carla]')
    parser.add_argument('--n_points', type=int, default=8192,
                        help='Point Number [default: 2048]')
    parser.add_argument('--n_iter', type=int, default=9)
    parser.add_argument('--act_steps', type=int, default=3)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay_steps', type=float, default=600000, metavar='M',
                        help='SGD momentum (default: 200000)')
    parser.add_argument('--decay_rate', type=float, default=0.7, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--dataset_path', type=str,
                        default='/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow2/', metavar='N',
                        help='dataset to use')
    parser.add_argument('--param_config', type=str, default='configs/config_train.yaml', metavar='N',
                        help='Dataset config file path')
    parser.add_argument('--model_dir', type=str, default='checkpoints/', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='load pretrained model for training')
    parser.add_argument('--rm_history', type=bool, default=False, metavar='N',
                        help='Whether to remove the history exp directories')
    parser.add_argument('--use_ot', type=bool, default=False, metavar='N',
                        help='Whether to estimtate the foreground motion fields')
    parser.add_argument('--step', type=int, default=100, metavar='S',
                        help='the interval of tensorboard logs(default: 100)')
    parser.add_argument('--n_workers', type=int, default=1, metavar='S',
                        help='the number of worker loaders (default: 1)')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--plot_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_files_num', type=int, default=282)
    # eval setting
    parser.add_argument('--eval_plot_freq', type=int, default=10)
    parser.add_argument('--random_dataset', action='store_true', default=False,
                        help='Whether to remove the history exp directories')
    parser.add_argument('--use_cluster', type=bool, default=False,
                        help='Whether to use the cluster') # 12032881
    args = parser.parse_args()



    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    use_fg_inds = True

    # root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow2/'
    train_dataset = CARLA3D(root_dir=args.dataset_path + 'train/', nb_points=args.n_points, mode="train",
                      use_fg_inds=use_fg_inds)
    test_dataset = CARLA3D(root_dir=args.dataset_path + 'val/', nb_points=args.n_points, mode="test",
                           use_fg_inds=use_fg_inds)

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = SceneFlowNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 51):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()

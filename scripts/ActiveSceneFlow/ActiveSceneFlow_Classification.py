"""
by Yinqi, 04, 26, 2023
你是一名高级的Python人工智能专家，善于构造网络进行训练。
你现在需要用PointNet2实现基于PyTorch构造网络并训练对三维点云的场景流进行分类。
你首先需要读取数据集代码，数据集格式如下：
1、数据集是一个文件夹路径，路径下面由train与val文件夹，其中train文件夹是训练集，val文件夹是测试集；
2、train文件夹个val文件夹里面的格式是一样的，包含了每辆车名字命名的文件夹，每辆车文件夹里面是连续的npz文件，每个npz文件包含了：
（1）gt.npy：车辆上一帧点云和当前帧点云之间的真实场景流结果，每个值代表着上一帧点云和当前帧点云对应点的平移向量，是大小为N*3的矩阵
（2）s_fg_mask.npy：是大小为N*1的矩阵，是这个场景流矩阵的语义分割结果，
语义分割0和1。

接着你的网络需要实现以下功能：
（1）s_fg_mask.npy是网络语义分割的lable，gt.npy是输入的数据
（2）使网络成为不因为点的数量变化而影响的网络
（3）需要每一次迭代完成后保存最新的和效果最好的model
（4）每次迭代需要以进度条的形式展现训练的进度
（5）如果训练中断了，可以后期读取最新model恢复训练的功能
（6）提供多个预设参数方便使用者改变
（7）每次输出结果后都要记录log
（8）写清楚注释
（9）描述一下你设计的网络结构并说清楚它的原理和优点
（10）可以支持多GPU训练
请总结以上代码，写出一个可用的代码文件，相关说明写在注释里面即可

用python写一个用于分类场景流的网络
场景流torch size输入大小为（B，N，3）
label大小为（B，N），其中只有0，1两个值

"""
import os
import numpy as np
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.datasets.my_carla import CARLA3D, Batch, add_Seg_Fuison, add_Seg_After
import argparse
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm # 可视化训练进度的工具
import os
from datetime import datetime

# 数据集类，继承自pytorch的Dataset类

class PointCloudDataset(Dataset):
    def __init__(self, data_path, split):
        assert split in ['train', 'val'], 'split should be either train or val'
        self.split = split
        self.data_path = data_path
        self.dataset = [] # 存储数据集中所有的点云场景流

        # 遍历数据集文件夹下的每个车辆文件夹
        for folder_name in os.listdir(os.path.join(data_path, split)):
            folder_path = os.path.join(data_path, split, folder_name)
            if os.path.isdir(folder_path):
                # 读取每辆车的点云场景流
                for npz_name in os.listdir(folder_path):
                    npz_path = os.path.join(folder_path, npz_name)
                    npz_file = np.load(npz_path)
                    pc = npz_file['gt']
                    label = npz_file['s_fg_mask']
                    self.dataset.append((pc, label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pc, label = self.dataset[idx]
        pc = torch.from_numpy(pc).float()
        label = torch.from_numpy(label).long()
        return pc, label

# 网络模型类
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


# class PointCloudClassifier(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
#         super(PointCloudClassifier, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(input_dim, hidden_dim, 1),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.ReLU(inplace=True)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 1),
#             nn.BatchNorm1d(hidden_dim * 4),
#             nn.ReLU(inplace=True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv1d(hidden_dim * 4, hidden_dim * 8, 1),
#             nn.BatchNorm1d(hidden_dim * 8),
#             nn.ReLU(inplace=True)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 8, hidden_dim * 4),
#             nn.BatchNorm1d(hidden_dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim * 4, hidden_dim * 2),
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim * 2, output_dim)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = torch.max(x, 2)[0]
#         out = self.classifier(x)
#         return out
# class PointNet(nn.Module):
#     def __init__(self, num_classes=8192):
#         super(PointNet, self).__init__()
#         # 初始化网络结构
#         self.num_classes = num_classes
#         self.conv1 = nn.Conv1d(3, 64, kernel_size=1, stride=1)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv1d(128, 256, kernel_size=1, stride=1)
#         self.relu3 = nn.ReLU()
#         self.global_max_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc1 = nn.Linear(256, 512)
#         self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         # 定义网络前向传播过程
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.global_max_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.relu4(x)
#         x = self.fc2(x)
#         return x
#
#
# class SceneFlowNet(nn.Module):
#     def __init__(self):
#         super(SceneFlowNet, self).__init__()
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.conv4 = nn.Conv1d(256, 512, 1)
#         self.conv5 = nn.Conv1d(512, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 2)
#
#     def forward(self, x):
#         x = x[:, :, :3] # 只保留前三个通道
#         x = x.permute(0, 2, 1) # permute维度使得x的形状为[batch_size, in_channels, length]
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = torch.max(x, dim=2)[0]
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# class SceneFlowClassifier(nn.Module):
#     def __init__(self):
#         super(SceneFlowClassifier, self).__init__()
#
#         self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1, stride=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1, stride=1)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1, stride=1)
#         self.bn5 = nn.BatchNorm1d(1024)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 2)
#
#     def forward(self, x):
#         # Input shape: (B, 3, N)
#         # Output shape: (B, 2)
#
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
#         x, _ = torch.max(x, 2)
#         x = F.relu(self.bn6(self.fc1(x)))
#         x = self.fc2(x)
#
#         return x

# class SceneFlowNet(torch.nn.Module):
#     def __init__(self):
#         super(SceneFlowNet, self).__init__()
#         self.conv1 = torch.nn.Conv1d(6, 64, kernel_size=3, stride=2, padding=1)
#         self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
#         self.conv5 = torch.nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1)
#         self.dropout = torch.nn.Dropout(p=0.5)
#         self.fc1 = torch.nn.Linear(1024, 512)
#         self.fc2 = torch.nn.Linear(512, 256)
#         self.fc3 = torch.nn.Linear(256, 2)
#
#     def forward(self, x):
#         x = torch.nn.functional.relu(self.conv1(x))
#         x = torch.nn.functional.relu(self.conv2(x))
#         x = torch.nn.functional.relu(self.conv3(x))
#         x = torch.nn.functional.relu(self.conv4(x))
#         x = torch.nn.functional.relu(self.conv5(x))
#         x = torch.nn.functional.dropout(x, p=0.5)
#         x = x.max(dim=-1)[0]
#         x = torch.nn.functional.relu(self.fc1(x))
#         x = torch.nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class PointNet(nn.Module):
    def __init__(self, n_points=8192):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(n_points)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(n_points)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(n_points)
        self.fc4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(n_points)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(n_points)
        self.fc6 = nn.Linear(256, n_points) # 分类目标是静态/动态

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.fc6(x)
        return x

# 训练函数
def train(model, device, train_loader, optimizer, epoch, log_file):
    model.train() # 设置模式为训练模式
    train_loss = 0
    correct = 0

    # 使用tqdm库展示训练进度
    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for batch_idx, data in enumerate(train_loader):
            flow = data['ground_truth'][1]
            mask = data['mask'][0]  # 12032881
            # flow = np.concatenate([flow, np.zeros((flow.shape[0], 3))], axis=1)
            # flow = flow.contiguous().cuda().float()

            # flow = flow.transpose(2, 1).contiguous()
            # mask = mask.transpose(1, 0).contiguous()
            # mask = mask.contiguous().cuda().float()
            flow, mask = flow.to(device), mask.to(device)
            # print(flow.size())
            # print(mask.size())
            # exit()
            optimizer.zero_grad() # 清空梯度
            output = model(flow) # 数据通过网络
            mask= torch.tensor(mask, dtype=torch.long)
            # output = torch.tensor(output, dtype=torch.long)
            # label = torch.tensor(label, dtype=torch.float)

            loss = nn.functional.cross_entropy(output, mask) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True) # 预测结果
            correct += pred.eq(mask.view_as(pred)).sum().item()
            pbar.update(1)

    # 计算并记录训练结果
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    log_file.write('Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.2f}%\n'.format(
        epoch, train_loss, accuracy))

# 测试函数
def test(model, device, test_loader, log_file):
    model.eval() # 设置模式为测试模式
    test_loss = 0
    correct = 0

    with torch.no_grad():
        with tqdm(enumerate(test_loader), total=len(test_loader)) as pbar:
            for batch_idx, data in enumerate(test_loader):
                flow = data['ground_truth'][1]
                mask = data['mask'][0]  # 12032881
                flow, mask = flow.to(device), mask.to(device)
                mask = torch.tensor(mask, dtype=torch.long)
                output = model(flow)
                test_loss += nn.functional.cross_entropy(output, mask, reduction='sum').item() # 损失累计
                pred = output.argmax(dim=3, keepdim=True)
                correct += pred.eq(output.view_as(pred)).sum().item()

    # 计算并记录测试结果
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    log_file.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy

if __name__ == '__main__':
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




    # 参数设置
    data_path = 'path/to/dataset'
    batch_size = 32
    epochs = 100
    lr = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 自动选择计算设备
    log_path = 'log.txt'
    best_model_path = 'classification_best_model.pth'
    latest_model_path = 'classification_latest_model.pth'

    # 定义数据集和数据加载器
    use_fg_inds = True
    train_dataset = CARLA3D(root_dir=args.dataset_path + 'train/', nb_points=args.n_points, mode="train",
                      use_fg_inds=use_fg_inds)
    val_dataset = CARLA3D(root_dir=args.dataset_path + 'val/', nb_points=args.n_points, mode="test",
                           use_fg_inds=use_fg_inds)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True,
        collate_fn=Batch, drop_last=True, timeout=0, persistent_workers=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True,
        collate_fn=Batch, drop_last=False, timeout=0, persistent_workers=True)

    # 创建网络实例并放到计算设备上
    model = PointNet(n_points = args.n_points).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 创建日志文件，记录训练结果
    log_file = open(log_path, 'a')
    log_file.write('=== Start Training: {} ===\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # 开始训练
    best_val_loss = float('inf')
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_file)
        val_loss, val_acc = test(model, device, val_loader, log_file)

        # 保存最新模型
        torch.save(model.state_dict(), latest_model_path)

        # 保存效果最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    log_file.write('=== End Training: {} ===\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log_file.write('Best Validation Loss: {:.4f}\tBest Validation Accuracy: {:.2f}%\n'.format(best_val_loss, best_val_acc))
    log_file.close()

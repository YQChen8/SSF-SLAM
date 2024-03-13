#! /home/zhijun/anaconda3/bin/python
"""
by Yinqi, 02, 10, 2023
"""

# 导入需要的库和数据
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Float64MultiArray
from pyquaternion import Quaternion
import glob
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import Counter


def slove_RT_by_SVD(src, dst):
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)

    src = src - src_mean  # n, 3
    dst = dst - dst_mean
    H = np.transpose(src) @ dst

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        # print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T & U.T

    t = -R @ src_mean.T + dst_mean.T  # 3, 1

    return R, t


if __name__ == '__main__':
    rospy.init_node('velodyne_points_odometry_node', anonymous=True)
    velodyne_points_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=100)
    odom_pub = rospy.Publisher('frame_odom1', Float64MultiArray, queue_size=100)  # 里程计话题发布者
    rate = rospy.Rate(10)  # 10hz
    rospy.loginfo("\033[1;32m----> PointCloudOdometry Started.\033[0m")

    # root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow/train/record2022_1210_2202/rm_road/SF/04'
    root_dir = '//scripts/FESTA/results'


    if rospy.has_param('~DATASET_PATH'):  # 是否存在参数
        root_dir = rospy.get_param('~DATASET_PATH')  # 获取参数

    filenames = []

    for sub_dir in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, sub_dir)
        filenames += glob.glob(sub_path)

    filenames.sort()

    for sub_dir in filenames:
        # 发布点云
        # print(sub_dir)
        rospy.loginfo(sub_dir)
        ac = np.load(sub_dir)
        points = ac['pos1']
        move_gt = ac['gt']
        # print(move_gt.shape)
        # move_gt = ac['flow']
        # move_gt_seg = ac['s_fg_mask']

        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = 'livox_frame'

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, np.float32).tobytes()

        velodyne_points_pub.publish(msg)

        ################################################################ 发布点云、里程计、轨迹、地图
        move_gt_addPC = np.concatenate((move_gt, points), axis=1)
        model = GaussianMixture(n_components=2)
        # model = KMeans(n_clusters=2)
        # 调用 fit 方法对数据进行聚类，先用高斯模糊算法，感觉效果比较好
        all_label = model.fit_predict(move_gt_addPC)  # 这个是所有分类
        bg_label = Counter(all_label).most_common(1)[0][0]  # 调用Counter函数找出出现最多的label作为背景label
        bg_index = np.argwhere(all_label == bg_label).flatten()  # 找出背景label所在的索引
        # print(bg_label_index.shape)
        # bg_index = bg_index[0: -1:5]
        # print(bg_label_index.shape)
        # all_label = np.ones_like(all_label)  # 更新所有分类数组
        # all_label[bg_label_index] = 0  # 把背景直接二分类，背景标为0
        # my_accuracy_score(move_gt_seg, all_label)

        # 获取里程计算R，t并发布
        # bg_index = np.argwhere(move_gt_seg == 0).flatten()

        target = points[bg_index] + move_gt[bg_index]
        source = points[bg_index]


        R, t = slove_RT_by_SVD(target, source)
        q = Quaternion(matrix=R)
        q = [q.x, q.y, q.z, q.w]
        t = t.flatten()

        para_t_q = np.hstack((t, q))
        para_t_q = Float64MultiArray(data=para_t_q)
        odom_pub.publish(para_t_q)
        # print(para_t_q)
        rate.sleep()

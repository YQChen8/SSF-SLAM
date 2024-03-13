#! /home/zhijun/anaconda3/bin/python
"""
by Yinqi, 02, 10, 2023
"""
import glob
import os
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Float64MultiArray
from pyquaternion import Quaternion


def slove_RT_by_SVD(src, dst):
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)

    src = src - src_mean  # n, 3
    dst = dst - dst_mean
    H = np.transpose(src) @ dst

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T & U.T

    t = -R @ src_mean.T + dst_mean.T  # 3, 1

    return R, t


if __name__ == '__main__':

    rospy.init_node('velodyne_points_odometry_node', anonymous=True)
    # velodyne_points_pub = rospy.Publisher('lidars_fusion/raw_data', PointCloud2, queue_size=100)
    velodyne_points_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=100)
    odom_pub = rospy.Publisher('frame_odom1', Float64MultiArray, queue_size=100)  # 里程计话题发布者
    rate = rospy.Rate(10)  # 10hz

    # root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow/train/record2022_1210_2202/rm_road/SF/02'
    # root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow/val/record2022_1211_0020/rm_road/SF/05'
    # root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow/train/record2022_1210_2202/rm_road/SF/11'
    root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow/train/record2022_1210_2202/rm_road/SF/04'
    if rospy.has_param('~DATASET_PATH'):  # 是否存在参数
        root_dir = rospy.get_param('~DATASET_PATH')  # 获取参数

    filenames = []

    for sub_dir in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, sub_dir)
        filenames += glob.glob(sub_path)

    filenames.sort()

    for sub_dir in filenames:
        # 发布点云
        print(sub_dir)
        ac = np.load(sub_dir)
        points = ac['pos1']
        move_gt = ac['gt']
        move_gt_seg = ac['s_fg_mask']

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
        msg.data = np.asarray(points, np.float32).tostring()

        velodyne_points_pub.publish(msg)

        # 获取里程计算R，t并发布
        bg_index = np.argwhere(move_gt_seg == 0).flatten()

        target = points[bg_index] + move_gt[bg_index]
        source = points[bg_index]

        R, t = slove_RT_by_SVD(target, source)
        q = Quaternion(matrix=R)
        q = [q.x, q.y, q.z, q.w]
        t = t.flatten()

        para_t_q = np.hstack((t, q))
        para_t_q = Float64MultiArray(data=para_t_q)
        odom_pub.publish(para_t_q)
        print(para_t_q)
        rate.sleep()

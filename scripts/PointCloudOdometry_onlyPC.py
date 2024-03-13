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

if __name__ == '__main__':

    rospy.init_node('velodyne_points_node', anonymous=True)
    velodyne_points_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    root_dir = '/home/zhijun/ISUS/public_dataset_nas/carla_scene_flow/train/record2022_1210_2202/rm_road/SF/14'
    if rospy.has_param('~DATASET_PATH'):  # 是否存在参数
        root_dir = rospy.get_param('~DATASET_PATH')  # 获取参数
    filenames = []
    for sub_dir in sorted(os.listdir(root_dir)):
        # if self.mode == "train" or self.mode == "val":
        sub_path = os.path.join(root_dir, sub_dir)
        # all_sub_paths = sorted(os.listdir(sub_path))
        # for sub_sub_dir in all_sub_paths:
        # pattern = sub_path + '/' + sub_sub_dir + "/" + "*.npz"
        # pattern = sub_path + "/" + "*.npz"
        # filenames += glob.glob(pattern)
        filenames += glob.glob(sub_path)

    filenames.sort()
    # for sub_dir in filenames:
    #     print(sub_dir)
    # exit()

    for sub_dir in filenames:
        # 发布点云
        print(sub_dir)
        ac = np.load(sub_dir)
        points = ac['pos1']
        print(points.shape)
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
        rate.sleep()

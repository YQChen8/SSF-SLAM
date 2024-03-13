import glob

root_dir = 'plt_results/00_8192_100_GPG/'  # 图片目录
output = 'plt_results/00_8192_100_GPG/result00_Pointnet2_Seg_8192_100_GPG.mp4'
import os
import cv2
from PIL import Image
import numpy as np


def image_to_video(image_path, media_path):
    '''
    图片合成视频函数
    :param image_path: 图片路径
    :param media_path: 合成视频保存路径
    :return:
    '''
    # 获取图片路径下面的所有图片名称
    image_names = os.listdir(image_path)
    # # print(image_names)
    # image_names = np.sort(image_names)
    # print(image_names)
    # image_names = []
    #
    # for sub_dir in sorted(os.listdir(root_dir)):
    #     sub_path = os.path.join(root_dir, sub_dir)
    #     image_names += glob.glob(sub_path)

    # image_names.sort()

    # 对提取到的图片名称进行排序
    image_names.sort(key=lambda n: int(n[:-4]))
    print(image_names)
    # 设置写入格式
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # 设置每秒帧数
    fps = 5  # 由于图片数目较少，这里设置的帧数比较低
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(root_dir+image_names[0])
    # 初始化媒体写入对象
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    # 遍历图片，将每张图片加入视频当中
    for image_name in image_names:
        im = cv2.imread(os.path.join(image_path, image_name))
        media_writer.write(im)
        print(image_name, '合并完成！')
    # 释放媒体写入对象
    media_writer.release()
    print('视频写入完成！')


image_to_video(root_dir, output)

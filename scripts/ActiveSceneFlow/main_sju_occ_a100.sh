#!/bin/bash
#SBATCH -o out_a100/job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=a100      # 作业提交的指定分区队列为titan
#SBATCH --qos=a100           # 指定作业的QOS
#SBATCH -J myFirstGPUJob       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=96    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制;

#python main_sju_occ.py --n_points=8192 --rm_history=True --batch_size=32 --n_workers=16 --exp_name='TFlow_titan4_8192_100' --dataset_path='../carla_scene_flow2/'
#nvidia-smi
#python main_sju_occ.py --n_points=8192 --rm_history=True --batch_size=20 --n_workers=4 --exp_name='TFlow_titan4_8192_100_new' --dataset_path='../carla_scene_flow2/'
#python main_sju_occ.py --use_cluster=True --n_points=8192 --pretrained --rm_history=True --batch_size=40 --n_workers=96 --exp_name='TFlow_a100_8192_4000_addSegafterPC0820_7dim' --dataset_path='../carla_scene_flow2/'

python main_sju_occ_a100.py --use_cluster=True --n_points=8192  --pretrained --rm_history=True --batch_size=20 --n_workers=96 --exp_name='TFlow_a100_8192_4000_addSegafterPC0824_7dim' --dataset_path='../carla_scene_flow2/'

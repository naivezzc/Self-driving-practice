from nuscenes.nuscenes import NuScenes
import numpy as np
import cv2
import os
from pyquaternion import Quaternion

# Build nuscenes class
version = "v1.0-mini"
data_root = '/home/zhang_hm/work/dataset/v1.0-mini'
nuscenes = NuScenes(version, data_root, verbose=False)

# Basic information
print("len: {}".format(len(nuscenes.sample)))
sample = nuscenes.sample[0]
print("sample keys: {}".format(sample.keys()))
# print("sample: {}".format(sample))

# Get lidar information
lidar_token = sample['data']['LIDAR_TOP']
lidar_sample_data = nuscenes.get('sample_data', lidar_token)
lidar_filename = os.path.join(data_root, lidar_sample_data['filename'])
print(lidar_sample_data)

# Load point cloud data
lidar_point = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 5)
print(lidar_point)

'''坐标系
    1.全局坐标系   global coordinate (车辆在t0时刻的位置为全局坐标系的原点)
    2.车体坐标系   ego coordinate （以车体为原点的坐标系）
    3.传感器坐标系 
        -lidar
        -radar
        -camera
    标定 calibrator
        -lidar的标定获得的结果是lidar相对于ego的位置（translation），和旋转（rotation）
            -translation 可以用3个float表示位置
            -rotation 用4个float表示旋转， 四元数
        -camera的标定获得的结果是camera相对于ego的位置（translation），和旋转（rotation
            -相机内参（intrinsics）
            -相机畸变（目前nuScense不考虑）
    不同传感器频率不同，捕获时间不同
    -lidar   捕获的timestamp是t0，t0->egopose
    -camera  捕获的timestamp是t1，t1->egopose
    lidar point —> camera
    lidar point -> egopose0 -> global -> egopose1 -> camera -> intrinsic -> image
'''

# lidar point -> egopose0
# lidar point是基于lidar coordinate 而言的
# lidar coordinate lidar_pose(基于ego)
# lidar_points = lidar_pose @ lidar_points
lidar_calibrator_data = nuscenes.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
print(lidar_calibrator_data)

# Quaternion : [1x4] -> [3x3]矩阵
print("Rotation matrix :{}".format(Quaternion(lidar_calibrator_data['rotation']).rotation_matrix))


def get_matrix(calibrated_data):
    quaternion, translation = calibrated_data['rotation'], calibrated_data['translation']
    output = np.eye(4)
    output[:3, :3] = Quaternion(quaternion)
    output[:3, 3] = translation
    return output


'''
lidar_pose是基于ego而言
point = lidar_pose @ lidar_points 代表了 lidar->ego的过程
'''
lidar2ego = get_matrix(lidar_calibrator_data)
print(lidar2ego)

'''
ego_pose0 -> global
ego_pose就是基于global坐标系的
ego_pose @ ego_points.T   ego->global
'''
ego_pose = nuscenes.get('ego_pose', lidar_sample_data['ego_pose_token'])
ego2global = get_matrix(ego_pose)
lidar2global = lidar2ego @ ego2global
print(ego2global)



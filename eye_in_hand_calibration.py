import cv2
import numpy as np
from spatial_transform import multiply_transform, invert_transform, get_rmtx, get_rvec, get_rvec_Yaskawa
from calib_utils import show_RT, invert_RT_list, load_camera_params, load_gripper2base, get_board2cam, generate_pointclouds_in_new_coordinate



robot_pose_filenames = []
color_image_filenames = []
depth_map_filenames = []

# 圆心距
center_distance=20
# 图片尺寸
width=1920
height=1200
# 标定组数
num=8

param_txt_path = "input_EyeInHand/param.txt"
output_path = "output_EyeInHand/"

camera_mtx, camera_dist = load_camera_params(param_txt_path)

for i in range(num):
    robot_pose_filenames.append(f'input_EyeInHand/pos{i}.txt')
    color_image_filenames.append(f'input_EyeInHand/pos{i}.bmp')
    depth_map_filenames.append(f'input_EyeInHand/pos{i}.tiff')

# 获取夹爪和机械臂底座的关系
gripper2base_rmtx_list, gripper2base_rvec_list, gripper2base_tvec_list = load_gripper2base(robot_pose_filenames,num)

# 获取相机和标定板的关系
board2cam_rmtx_list, board2cam_tvec_list = get_board2cam(color_image_filenames, depth_map_filenames, 
                                                         camera_mtx, camera_dist,center_distance,num)

print('show gripper2base')
show_RT(gripper2base_rmtx_list, gripper2base_tvec_list)

print('show board2cam')
show_RT(board2cam_rmtx_list, board2cam_tvec_list)

# 计算出相机和夹爪的关系
# Refer to the opencv document for more details
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
cam2gripper_rmtx, cam2gripper_tvec = cv2.calibrateHandEye(gripper2base_rmtx_list, gripper2base_tvec_list, board2cam_rmtx_list, board2cam_tvec_list,
                                                    method=cv2.CALIB_HAND_EYE_TSAI)

cam2gripper_rmtx = np.matrix(cam2gripper_rmtx)
cam2gripper_tvec = np.matrix(cam2gripper_tvec)

print('Cam2gripper的数据如下(R、T):')
print("Cam2gripper R:", cam2gripper_rmtx)
print("Cam2gripper T:", cam2gripper_tvec)

cam2gripper_rvec = get_rvec(cam2gripper_rmtx)

#通过cam2gripper 和 gripper2base，计算cam2base
cam2base_rvec_list = []
cam2base_tvec_list = []
for i in range(num):
    gripper2base_rvec = gripper2base_rvec_list[i]
    gripper2base_tvec = gripper2base_tvec_list[i]

    cam2base_rvec, cam2base_tvec = multiply_transform(gripper2base_rvec, gripper2base_tvec, cam2gripper_rvec, cam2gripper_tvec)
    cam2base_rvec_list.append(cam2base_rvec)
    cam2base_tvec_list.append(cam2base_tvec)

# 通过cam2base，将点云旋转到base坐标系
generate_pointclouds_in_new_coordinate(camera_mtx, camera_dist, 
                                       color_image_filenames, depth_map_filenames, robot_pose_filenames, 
									   cam2base_rvec_list, cam2base_tvec_list, width, height, output_path,num)


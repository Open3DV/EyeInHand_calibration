import cv2
import numpy as np
from spatial_transform import multiply_transform, invert_transform, get_rmtx, get_rvec, get_rvec_Yaskawa
from calib_utils import show_RT, invert_RT_list, load_camera_params, load_gripper2base, get_board2cam, generate_pointclouds_in_new_coordinate



pos_txt = []
bright = []
depth = []

# 圆心距
center_distance = 40
# 图片尺寸
width = 1022
height = 768
# 标定组数
num = 5

param_txt_path = "input_EyeToHand/param.txt"
output_path = "output_EyeToHand/"

camera_mtx, camera_dist = load_camera_params(param_txt_path)

for i in range(num):
    pos_txt.append(f'input_EyeToHand/pos{i}.txt')
    bright.append(f'input_EyeToHand/pos{i}.bmp')
    depth.append(f'input_EyeToHand/pos{i}.tiff')

# 获取夹爪和机械臂底座的关系
gripper2base_rmtx_list, gripper2base_rvec_list, gripper2base_tvec_list = load_gripper2base(pos_txt,num)

# gripper2base求逆得到base2gripper
base2gripper_rmtx_list, base2gripper_tvec_list = invert_RT_list(gripper2base_rmtx_list, gripper2base_tvec_list)

# 获取相机和标定板的关系
board2cam_rmtx_list, board2cam_tvec_list = get_board2cam(bright, depth, camera_mtx, camera_dist,center_distance,num)

print('show gripper2base')
show_RT(gripper2base_rmtx_list, gripper2base_tvec_list)

print('show board2cam')
show_RT(board2cam_rmtx_list, board2cam_tvec_list)

# 计算出相机和机械臂底座的关系
# Refer to the opencv document for more details
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
cam2base_rmtx, cam2base_tvec = cv2.calibrateHandEye(base2gripper_rmtx_list, base2gripper_tvec_list, board2cam_rmtx_list, board2cam_tvec_list,
                                                    method=cv2.CALIB_HAND_EYE_TSAI)

cam2base_rmtx = np.matrix(cam2base_rmtx)
cam2base_tvec = np.matrix(cam2base_tvec)

print('Cam2Base的数据如下(R、T)：')
print("Cam2Base R:", cam2base_rmtx)
print("Cam2Base T:", cam2base_tvec)

cam2base_rvec = get_rvec(cam2base_rmtx)

#通过 cam2base 和 base2gripper，计算cam2gripper
cam2gripper_rvec_list = []
cam2gripper_tvec_list = []
for i in range(num):
    base2gripper_rmtx = base2gripper_rmtx_list[i]
    base2gripper_tvec = base2gripper_tvec_list[i]
    base2gripper_rvec = get_rvec(base2gripper_rmtx)

    cam2gripper_rvec, cam2gripper_tvec = multiply_transform(base2gripper_rvec, base2gripper_tvec, cam2base_rvec, cam2base_tvec)
    cam2gripper_rvec_list.append(cam2gripper_rvec)
    cam2gripper_tvec_list.append(cam2gripper_tvec)

# 通过cam2gripper，将点云旋转到gripper坐标系
generate_pointclouds_in_new_coordinate(camera_mtx, camera_dist, bright, depth, pos_txt, cam2gripper_rvec_list, cam2gripper_tvec_list, width, height, output_path,num, depth_ratio=1000)


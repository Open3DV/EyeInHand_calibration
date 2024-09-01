import cv2
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from draw_tools import draw_axis, set_axes_equal
from spatial_transform import multiply_transform, invert_transform, get_rmtx, get_rvec, get_rvec_Yaskawa


def show_RT(rmtxs, tvecs):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    draw_axis(ax, rvec=np.matrix([[.0,.0,.0]]).T, tvec=np.matrix([[.0,.0,.0]]).T)

    for i in range(len(rmtxs)):
        rmtx = rmtxs[i]
        rvec = get_rvec(rmtx)
        tvec = tvecs[i]

        draw_axis(ax, rvec=rvec, tvec=tvec, index=i)

    set_axes_equal(ax)
    plt.show()
    
def invert_RT_list(rmtx_list, tvec_list):
    rmtx_inv_list = []
    tvec_inv_list = []
    for i in range(len(rmtx_list)):
        rmtx = rmtx_list[i]
        tvec = tvec_list[i]
        rvec = get_rvec(rmtx)
        rvec_inv, tvec_inv = invert_transform(rvec, tvec)
        rmtx_inv = get_rmtx(rvec_inv)
        #rmtx_inv, tvec_inv = InvTransformRT(rmtx, tvec)
        rmtx_inv_list.append(rmtx_inv)
        tvec_inv_list.append(tvec_inv)
    return rmtx_inv_list, tvec_inv_list



def load_depth_map(depth_tiff_file):

    depth = cv2.imread(depth_tiff_file, -1)
    depth = np.float32(np.array(depth))

    return depth

def load_camera_params(params_file):
    params = np.loadtxt(params_file)
    camera_mtx = np.zeros((3, 3))
    camera_mtx[0, 0] = params[0]
    camera_mtx[0, 2] = params[2]
    camera_mtx[1, 1] = params[4]
    camera_mtx[1, 2] = params[5]
    camera_mtx[2, 2] = 1
    camera_dist = params[9:14]

    return camera_mtx, camera_dist

def load_gripper2base(pos_txt,num):

    tcp2base_rmtx_list = []
    tcp2base_rvec_list = []
    tcp2base_tvec_list = []

    print('txt')
    print(pos_txt)
    for i in range(num):
        data = pos_txt[i]
        f = open(data, "r")
        lines = f.readlines()  # 读取全部内容
        list1 = []
        for line in lines:
            list1.append(line.strip().split('\t'))
        rvec = get_rvec_Yaskawa(list1[0][0], list1[1][0], list1[2][0])
        tvec = np.matrix([list1[3][0], list1[4][0], list1[5][0]], dtype=np.float32).T
        rmtx = get_rmtx(rvec)
        tcp2base_rvec_list.append(rvec)
        tcp2base_rmtx_list.append(rmtx)
        tcp2base_tvec_list.append(tvec)

    return tcp2base_rmtx_list, tcp2base_rvec_list, tcp2base_tvec_list


def get_board2cam(bright,depth, camera_mtx, camera_dist,center_distance,num):
    board2cam_rmtxs = []
    board2cam_tvecs = []

    for i in range(num):
        color_img_path = bright[i]
        depth_img_path = depth[i]

        #rmtx, tvec = get_board2cam_Transform_2D(color_img_path, camera_mtx, camera_dist,center_distance)
        objectpoints = generate_calibration_board_points(11, 7, center_distance,center_distance)
        color_img = cv2.imread(color_img_path, 0)
        color_img = 255 - color_img
        ret, centers = cv2.findCirclesGrid(color_img, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        _, rvec, tvec = cv2.solvePnP(objectpoints, centers, camera_mtx, camera_dist)
        rmtx = get_rmtx(np.matrix(rvec))
        tvec = np.matrix(tvec)

        board2cam_rmtxs.append(rmtx)
        board2cam_tvecs.append(tvec)

    return board2cam_rmtxs, board2cam_tvecs


def generate_calibration_board_points(xNum, yNum, xSpace, ySpace):
    objectpoints = []
    for j in range(xNum):
        for i in range(yNum):
            if j % 2 == 0:
                x = i * xSpace
            else:
                x = (i + 0.5) * xSpace
            y = j * ySpace * 0.5
            z = 0
            objectpoint = np.array([x, y, z])
            objectpoints.append(objectpoint)
    objectpoints = np.array(objectpoints)

    return objectpoints

def generate_pointclouds_in_new_coordinate(camera_mtx, camera_dist, bright, depth, pos_txt, cam2new_rvec_list, cam2new_tvec_list, width, height, output, num, depth_ratio=1):

    print("生成点云")
    print(bright)
    print(depth)
    bright_all = bright
    depth_all = depth

    for i in range(num):
        print(i)
        cam2new_rvec = cam2new_rvec_list[i]
        cam2new_rmtx = get_rmtx(cam2new_rvec)
        cam2new_tvec = cam2new_tvec_list[i]

        color_img_path = bright_all[i]
        depth_img_path = depth_all[i]

        color = cv2.imread(color_img_path)
        depth = load_depth_map(depth_img_path)
        pointcloud = get_pointcloud_from_depthmap(depth*depth_ratio, color, camera_mtx, camera_dist,width,height)

        # 通过计算得到需要的参数
        pointcloud_new = np.zeros_like(pointcloud)
        pts = pointcloud[:, :3]
        color = pointcloud[:, 3:]
        #cam_pos_rmtx, cam_pos_tvec = Get_Camera_Position(cam2tcp_rmtx, cam2tcp_tvec, tcp2base_rmtx, tcp2base_tvec)

        # 计算得到需要的点云
        pts = np.matrix(pts).T
        pts = cam2new_rmtx * pts + cam2new_tvec
        pts = np.array(pts).T
        pointcloud_new[:, :3] = pts
        pointcloud_new[:, 3:] = color
        list=['pos0','pos1','pos2','pos3','pos4','pos5','pos6','pos7',]
        np.savetxt(os.path.join(output, list[i] + '_base_new.xyz'), pointcloud_new, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])

    return 0


def get_pointcloud_from_depthmap(depth, color, camera_mtx, camera_dist, width, height):

    ix, iy = np.meshgrid(range(width), range(height))
    ix = ix.flatten()
    iy = iy.flatten()

    depth = depth.flatten()
    color = color.reshape(-1, 3)
    ixiy = np.array(np.vstack((ix, iy)).T, dtype=np.float32)
    
    ixiy = cv2.undistortPoints(ixiy, camera_mtx, camera_dist)
    
    z = depth
    x = ixiy[:, 0, 0] * z
    y = ixiy[:, 0, 1] * z
    pointcloud = np.vstack((x, y, z)).T
    pointcloud = np.hstack((pointcloud, color))

    # remove invalid points
    pointcloud = pointcloud[pointcloud[:, 2] > 0]

    return pointcloud

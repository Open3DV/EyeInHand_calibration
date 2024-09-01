import cv2
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from draw_tools import draw_axis, set_axes_equal
from spatial_transform import multiply_transform, invert_transform

def calib_EyeInHand():
    pos_txt = []
    bright = []
    depth = []

    #圆心距
    center_distance=20
    #图片尺寸
    width=1920
    height=1200
    #标定组数
    num=8

    param_txt_path = "input_EyeInHand/param.txt"
    output_path = "output_EyeInHand/"

    camera_mtx, camera_dist = load_camera_params(param_txt_path)

    for i in range(num):
        pos_txt.append(f'input_EyeInHand/pos{i}.txt')
        bright.append(f'input_EyeInHand/pos{i}.bmp')
        depth.append(f'input_EyeInHand/pos{i}.tiff')

    # 获取夹爪和机械臂底座的关系
    tcp2base_rmtx_list, tcp2base_rvec_list, tcp2base_tvec_list = Get_TCP2Base(pos_txt,num)

    # 获取相机和标定板的关系
    board2cam_rmtxs, board2cam_tvecs = Get_Board2Cam(bright, depth, camera_mtx, camera_dist,center_distance,num,use_2D=True)

    # 计算出相机和夹爪的关系
    cam2tcp_rmtx, cam2tcp_tvec = cv2.calibrateHandEye(tcp2base_rmtx_list, tcp2base_tvec_list, board2cam_rmtxs, board2cam_tvecs,
                                                      method=cv2.CALIB_HAND_EYE_TSAI)
    cam2tcp_rmtx = np.matrix(cam2tcp_rmtx)
    cam2tcp_tvec = np.matrix(cam2tcp_tvec)

    print('Cam2Tcp的数据如下(R、T):')
    print("Cam2Tcp R:", cam2tcp_rmtx)
    print("Cam2Tcp T:", cam2tcp_tvec)

    cam2tcp_rvec = get_rvec(cam2tcp_rmtx)
    
    #通过cam2tcp和tcp2base，计算cam2base
    cam2base_rvec_list = []
    cam2base_tvec_list = []
    for i in range(num):
        tcp2base_rvec = tcp2base_rvec_list[i]
        tcp2base_tvec = tcp2base_tvec_list[i]

        cam2base_rvec, cam2base_tvec = multiply_transform(tcp2base_rvec, tcp2base_tvec, cam2tcp_rvec, cam2tcp_tvec)
        cam2base_rvec_list.append(cam2base_rvec)
        cam2base_tvec_list.append(cam2base_tvec)

    # 通过cam2base，将点云旋转到base坐标系
    generate_pointclouds_in_new_coordinate(camera_mtx, camera_dist, bright, depth, pos_txt, cam2base_rvec_list, cam2base_tvec_list, width, height, output_path,num)


def calib_EyeToHand():
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
    print(camera_mtx)
    print(camera_dist)

    for i in range(num):
        pos_txt.append(f'input_EyeToHand/pos{i}.txt')
        bright.append(f'input_EyeToHand/pos{i}.bmp')
        depth.append(f'input_EyeToHand/pos{i}.tiff')

    gripper2base_rmtx_list, gripper2base_rvec_list, gripper2base_tvec_list = Get_TCP2Base(pos_txt,num)

    cam2base_rmtx, cam2base_tvec=calibrateEyeToHand(camera_mtx, camera_dist, bright, depth, pos_txt,center_distance,output_path,num)
    cam2base_rvec = get_rvec(cam2base_rmtx)

    #通过 cam2gripper 和 gripper2base，计算cam2base
    cam2gripper_rvec_list = []
    cam2gripper_tvec_list = []
    for i in range(num):
        gripper2base_rvec = gripper2base_rvec_list[i]
        gripper2base_tvec = gripper2base_tvec_list[i]

        #求逆
        base2gripper_rvec, base2gripper_tvec = invert_transform(gripper2base_rvec, gripper2base_tvec)

        cam2gripper_rvec, cam2gripper_tvec = multiply_transform(base2gripper_rvec, base2gripper_tvec, cam2base_rvec, cam2base_tvec)
        cam2gripper_rvec_list.append(cam2gripper_rvec)
        cam2gripper_tvec_list.append(cam2gripper_tvec)

    # 通过cam2gripper，将点云旋转到gripper坐标系
    generate_pointclouds_in_new_coordinate(camera_mtx, camera_dist, bright, depth, pos_txt, cam2gripper_rvec_list, cam2gripper_tvec_list, width, height, output_path,num, depth_ratio=1000)



def calibrateEyeInHand(camera_mtx, camera_dist, bright, depth, pos_txt, center_distance,output,num):

    tcp2base_rmtxs, tcp2base_rvecs, tcp2base_tvecs = Get_TCP2Base(pos_txt,num)
    # print(tcp2base_rmtxs)
    # print(tcp2base_tvecs)
    # assert 0

    board2cam_rmtxs, board2cam_tvecs = Get_Board2Cam(bright,depth, camera_mtx, camera_dist,center_distance,num,use_2D=True)
    # print(board2cam_rmtxs)
    # print(board2cam_tvecs)
    # assert 0

    cam2tcp_rmtx, cam2tcp_tvec = cv2.calibrateHandEye(tcp2base_rmtxs, tcp2base_tvecs, board2cam_rmtxs, board2cam_tvecs,
                                                      method=cv2.CALIB_HAND_EYE_TSAI)
    cam2tcp_rmtx = np.matrix(cam2tcp_rmtx)
    cam2tcp_tvec = np.matrix(cam2tcp_tvec)

    print('Cam2Tcp的数据如下(R、T):')
    print("Cam2Tcp R:", cam2tcp_rmtx)
    print("Cam2Tcp T:", cam2tcp_tvec)

    data=[]
    for i in range(3):
        for j in range(3):
            data.append(round(cam2tcp_rmtx[i, j],8))
    for g in range(3):
            data.append(round(cam2tcp_tvec[g, 0],8))

    file_path = os.path.join(os.getcwd(), output, 'result.txt')
    with open(file_path, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

    return cam2tcp_rmtx, cam2tcp_tvec


def calibrateEyeToHand(camera_mtx, camera_dist, bright, depth, pos_txt,center_distance,output_path,num):

    #base2tcp_rmtxs, base2tcp_tvecs = Get_Base2TCP(pos_txt,num)
    tcp2base_rmtxs, tcp2base_rvecs, tcp2base_tvecs = Get_TCP2Base(pos_txt,num)

    base2tcp_rmtxs, base2tcp_tvecs = invert_RT_list(tcp2base_rmtxs, tcp2base_tvecs)

    board2cam_rmtxs, board2cam_tvecs = Get_Board2Cam(bright, depth, camera_mtx, camera_dist,center_distance,num,use_2D=True)

    print('show tcp2base')
    show_RT(tcp2base_rmtxs, tcp2base_tvecs)

    print('show board2cam')
    show_RT(board2cam_rmtxs, board2cam_tvecs)

    # Refer to the opencv document for more details
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    cam2base_rmtx, cam2base_tvec = cv2.calibrateHandEye(base2tcp_rmtxs, base2tcp_tvecs, board2cam_rmtxs, board2cam_tvecs,
                                                      method=cv2.CALIB_HAND_EYE_TSAI)

    cam2base_rmtx = np.matrix(cam2base_rmtx)
    cam2base_tvec = np.matrix(cam2base_tvec)

    print('Cam2Base的数据如下(R、T)：')
    print("Cam2Base R:", cam2base_rmtx)
    print("Cam2Base T:", cam2base_tvec)

    data=[]
    for i in range(3):
        for j in range(3):
            data.append(round(cam2base_rmtx[i, j],8))
    for g in range(3):
            data.append(round(cam2base_tvec[g, 0],8))

    file_path = os.path.join(os.getcwd(), output_path, 'result.txt')
    with open(file_path, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

    return cam2base_rmtx, cam2base_tvec




def show_RT(rmtxs, tvecs):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    draw_axis(ax, rvec=np.matrix([[.0,.0,.0]]).T, tvec=np.matrix([[.0,.0,.0]]).T)

    for i in range(len(rmtxs)):
        rmtx = rmtxs[i]
        check_rmtx(rmtx)
        rvec = get_rvec(rmtx)
        tvec = tvecs[i]
        check_tvec(tvec)

        draw_axis(ax, rvec=rvec, tvec=tvec, index=i)

    set_axes_equal(ax)
    plt.show()
    
def invert_RT_list(rmtx_list, tvec_list):
    rmtx_inv_list = []
    tvec_inv_list = []
    for i in range(len(rmtx_list)):
        rmtx = rmtx_list[i]
        tvec = tvec_list[i]
        rmtx_inv, tvec_inv = InvTransformRT(rmtx, tvec)
        rmtx_inv_list.append(rmtx_inv)
        tvec_inv_list.append(tvec_inv)
    return rmtx_inv_list, tvec_inv_list


def get_rmtx(rvec):
    check_rvec(rvec)
    rmtx, _ = cv2.Rodrigues(rvec)
    rmtx = np.matrix(rmtx)
    check_rmtx(rmtx)
    return rmtx



def get_rvec(rmtx):
    check_rmtx(rmtx)
    rvec, _ = cv2.Rodrigues(rmtx)
    rvec = np.matrix(rvec)
    check_rvec(rvec)
    return rvec

def InvTransformRT(R, T):
    check_rmtx(R)
    check_tvec(T)

    R_inv = R.T
    T_inv = -R_inv * T

    return R_inv, T_inv

def check_tvec(tvec):
    assert tvec.__class__.__name__ == 'matrix'
    assert tvec.ndim == 2
    assert tvec.shape == (3, 1)
    assert tvec.dtype == np.float32 or tvec.dtype == np.float64

def check_rvec(rvec):
    assert rvec.__class__.__name__ == 'matrix'
    assert rvec.ndim == 2
    assert rvec.shape == (3, 1)
    assert rvec.dtype == np.float32 or rvec.dtype == np.float64

def check_rmtx(rmtx):
    assert rmtx.__class__.__name__ == 'matrix'
    assert rmtx.ndim == 2
    assert rmtx.shape == (3, 3)
    assert rmtx.dtype == np.float32 or rmtx.dtype == np.float64

def get_rvec_Yaskawa(rx, ry, rz):
    euler_angle = [rx, ry, rz]
    rmtx = R.from_euler('xyz', euler_angle, degrees=True).as_matrix()
    rmtx = np.matrix(rmtx, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(rmtx)
    rvec = np.matrix(rvec)

    check_rvec(rvec)
    return rvec

def load_depth_map(depth_tiff_file):

    depth = cv2.imread(depth_tiff_file, -1)
    depth = np.float32(np.array(depth))

    return depth

def Get_Board2Cam_Transform_2D(color_img_path, camera_mtx, camera_dist,center_distance):
    # print("Using 2D-3D to compute the Transform Matrix between Board and Camera")
    objectpoints = generate_calibration_board_points(11, 7, center_distance,center_distance)
    color_img = cv2.imread(color_img_path, 0)
    color_img = 255 - color_img
    ret, centers = cv2.findCirclesGrid(color_img, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    _, rvec, tvec = cv2.solvePnP(objectpoints, centers, camera_mtx, camera_dist)
    rmtx = get_rmtx(np.matrix(rvec))
    tvec = np.matrix(tvec)
    check_rmtx(rmtx)
    check_tvec(tvec)

    return rmtx, tvec


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

def Get_TCP2Base(pos_txt,num):

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


def Get_Board2Cam(bright,depth, camera_mtx, camera_dist,center_distance,num, use_2D=False, use_3D=False):
    board2cam_rmtxs = []
    board2cam_tvecs = []

    for i in range(num):
        color_img_path = bright[i]
        depth_img_path = depth[i]

        rmtx, tvec = Get_Board2Cam_Transform_2D(color_img_path, camera_mtx, camera_dist,center_distance)

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


def get_pointcloud_from_depthmap(depth, color, camera_mtx, camera_dist,width,height):

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


def Get_Camera_Position(cam2tcp_rmtx, cam2tcp_tvec, tcp2base_rmtx, tcp2base_tvec):
    cam_pos_rmtx = tcp2base_rmtx * cam2tcp_rmtx
    cam_pos_tvec = tcp2base_rmtx * cam2tcp_tvec + tcp2base_tvec
    return cam_pos_rmtx, cam_pos_tvec


if __name__ == '__main__':
    #EyeInHand
    calib_EyeInHand()

    #EyeToHand
    calib_EyeToHand()

